use std::{collections::HashMap, sync::Arc};

use ai00_core::{FinishReason, GenerateRequest, InputState, ThreadRequest, Token, TokenCounter};
use derivative::Derivative;
use futures_util::StreamExt;
use salvo::{
    oapi::{extract::JsonBody, ToResponse, ToSchema},
    prelude::*,
    sse::SseEvent,
    Depot, Writer,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;

use super::*;
use crate::{
    api::request_info,
    types::{Array, ThreadSender},
    SLEEP,
};

#[derive(Debug, Derivative, Deserialize, ToSchema)]
#[derivative(Default)]
#[serde(default)]
#[salvo(schema(
    example = json!({
        "prompt": [
            "The Eiffel Tower is located in the city of"
        ],
        "stop": [
            "\n\n",
            "."
        ],
        "stream": false,
        "max_tokens": 1000,
        "sampler": {
            "type": "Nucleus",
            "top_p": 0.5,
            "top_k": 128,
            "temperature": 1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "penalty_decay": 0.99654026
        },
        "state": "00000000-0000-0000-0000-000000000000"
    })
))]
struct CompletionRequest {
    prompt: Array<String>,
    state: InputState,
    #[derivative(Default(value = "256"))]
    max_tokens: usize,
    #[derivative(Default(value = "Array::Item(\"\\n\\n\".into())"))]
    stop: Array<String>,
    stream: bool,
    #[serde(alias = "logit_bias")]
    bias: HashMap<u32, f32>,
    bnf_schema: Option<String>,
    #[serde(alias = "sampler_override")]
    sampler: Option<SamplerParams>,
    #[derivative(Default(value = "0.5"))]
    top_p: f32,
    #[derivative(Default(value = "128"))]
    top_k: usize,
    #[derivative(Default(value = "1.0"))]
    temperature: f32,
}

impl CompletionRequest {
    fn to_generate_request(&self, prompt: String) -> GenerateRequest {
        let stop: Vec<String> = self.stop.clone().into();
        let bias = Arc::new(self.bias.clone());
        let sampler = match &self.sampler {
            Some(sampler) => sampler.clone().into(),
            None => SamplerParams::Nucleus(NucleusParams {
                top_p: self.top_p,
                top_k: self.top_k,
                temperature: self.temperature,
                ..Default::default()
            })
            .into(),
        };
        let state = self.state.clone().into();

        GenerateRequest {
            prompt,
            max_tokens: self.max_tokens,
            stop,
            sampler,
            bias,
            bnf_schema: self.bnf_schema.clone(),
            state,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "text_completion",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "text": " Paris, France",
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt": 11,
            "completion": 4,
            "total": 15,
            "duration": {
                "secs": 0,
                "nanos": 260801800
            }
        }
    })
))]
struct CompletionResponse {
    object: String,
    model: String,
    choices: Vec<CompletionChoice>,
    #[serde(rename = "usage")]
    counter: TokenCounter,
}

#[derive(Debug, Derivative, Serialize, ToSchema, ToResponse)]
#[derivative(Default)]
#[serde(rename_all = "snake_case")]
enum PartialCompletionRecord {
    Content(String),
    #[derivative(Default)]
    #[serde(untagged)]
    None(HashMap<String, String>),
}

#[derive(Debug, Default, Serialize, ToSchema, ToResponse)]
struct PartialCompletionChoice {
    delta: PartialCompletionRecord,
    index: usize,
    finish_reason: FinishReason,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
#[salvo(schema(
    example = json!({
        "object": "text_completion.chunk",
        "model": "assets/models\\RWKV-x060-World-3B-v2.1-20240417-ctx4096.st",
        "choices": [
            {
                "delta": {
                    "content": " Paris"
                },
                "index": 0,
                "finish_reason": null
            }
        ]
    })
))]
struct PartialCompletionResponse {
    object: String,
    model: String,
    choices: Vec<PartialCompletionChoice>,
}

async fn respond_one(depot: &mut Depot, request: CompletionRequest, res: &mut Response) {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let prompts: Vec<String> = Vec::from(request.prompt);
    let tokenizer = info.tokenizer;

    let mut set = JoinSet::new();
    for (index, prompt) in prompts.into_iter().enumerate() {
        let req = request.to_generate_request(prompt);
        let sender = sender.clone();
        let tokenizer = tokenizer.clone();
        set.spawn(async move {
            let (token_sender, token_receiver) = flume::unbounded();
            let _ = sender.send(ThreadRequest::Generate {
                request: Box::new(req),
                tokenizer,
                sender: token_sender,
            });

            let mut finish_reason = FinishReason::Null;
            let mut text = String::new();
            let mut stream = token_receiver.into_stream();

            while let Some(token) = stream.next().await {
                match token {
                    Token::Start => {}
                    Token::Content(token) => {
                        text += &token;
                    }
                    Token::Stop(reason, _) => {
                        finish_reason = reason;
                        break;
                    }
                    _ => unreachable!(),
                }
            }

            CompletionChoice {
                text,
                index,
                finish_reason,
            }
        });
    }

    let mut choices = Vec::new();
    while let Some(result) = set.join_next().await {
        if let Ok(choice) = result {
            choices.push(choice);
        }
    }
    choices.sort_by_key(|c| c.index);

    let json = Json(CompletionResponse {
        object: "text_completion".into(),
        model: model_name,
        choices,
        counter: TokenCounter::default(),
    });
    res.render(json);
}

async fn respond_stream(depot: &mut Depot, request: CompletionRequest, res: &mut Response) {
    let sender = depot.obtain::<ThreadSender>().unwrap();
    let info = request_info(sender.clone(), SLEEP).await;
    let model_name = info.reload.model_path.to_string_lossy().into_owned();

    let prompts: Vec<String> = Vec::from(request.prompt);
    let tokenizer = info.tokenizer;

    let (tx, rx) = flume::unbounded::<(usize, Token)>();

    for (index, prompt) in prompts.into_iter().enumerate() {
        let req = request.to_generate_request(prompt);
        let sender = sender.clone();
        let tokenizer = tokenizer.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let (token_sender, token_receiver) = flume::unbounded();
            let _ = sender.send(ThreadRequest::Generate {
                request: Box::new(req),
                tokenizer,
                sender: token_sender,
            });

            let mut stream = token_receiver.into_stream();
            while let Some(token) = stream.next().await {
                if tx.send((index, token)).is_err() {
                    break;
                }
            }
        });
    }

    let stream = rx.into_stream().map(move |(index, token)| {
        let choice = match token {
            Token::Content(token) => PartialCompletionChoice {
                delta: PartialCompletionRecord::Content(token),
                index,
                ..Default::default()
            },
            Token::Stop(finish_reason, _) => PartialCompletionChoice {
                index,
                finish_reason,
                ..Default::default()
            },
            Token::Done => return Ok(SseEvent::default().text("[DONE]")),
            _ => return Ok(SseEvent::default()),
        };

        match serde_json::to_string(&PartialCompletionResponse {
            object: "text_completion.chunk".into(),
            model: model_name.clone(),
            choices: vec![choice],
        }) {
            Ok(json_text) => Ok(SseEvent::default().text(json_text)),
            Err(err) => Err(err),
        }
    });
    salvo::sse::stream(res, stream);
}

/// Generate completions for the given text.
#[endpoint(
    responses(
        (status_code = 200, description = "Generate one response if `stream` is false.", body = CompletionResponse),
        (status_code = 201, description = "Generate SSE response if `stream` is true", body = PartialCompletionResponse)
    )
)]
pub async fn completions(depot: &mut Depot, req: JsonBody<CompletionRequest>, res: &mut Response) {
    let request = req.0;
    match request.stream {
        true => respond_stream(depot, request, res).await,
        false => respond_one(depot, request, res).await,
    }
}
