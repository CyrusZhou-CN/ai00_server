use axum::{extract::State, Json};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{GenerateRequest, OptionArray, ThreadRequest, ThreadState, Token, TokenCounter};

#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingRequest {
    pub input: OptionArray<String>,
}

impl From<EmbeddingRequest> for GenerateRequest {
    fn from(value: EmbeddingRequest) -> Self {
        Self {
            prompt: Vec::from(value.input).join(""),
            max_tokens: 1,
            embed: true,
            ..Default::default()
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<EmbeddingData>,
    #[serde(rename = "usage")]
    pub counter: TokenCounter,
}

pub async fn embeddings(
    State(ThreadState {
        sender, model_name, ..
    }): State<ThreadState>,
    Json(request): Json<EmbeddingRequest>,
) -> Json<EmbeddingResponse> {
    let (token_sender, token_receiver) = flume::unbounded();
    let model_name = model_name.read().unwrap().clone();

    let _ = sender.send(ThreadRequest::Generate {
        request: request.into(),
        token_sender,
    });

    let mut token_counter = TokenCounter::default();
    let mut embedding = Vec::new();
    let mut stream = token_receiver.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Stop(_, counter) => token_counter = counter,
            Token::Embed(emb) => embedding = emb,
            Token::Done => break,
            _ => {}
        }
    }

    Json(EmbeddingResponse {
        object: "list".into(),
        model: model_name,
        data: vec![EmbeddingData {
            object: "embedding".into(),
            index: 0,
            embedding,
        }],
        counter: token_counter,
    })
}
