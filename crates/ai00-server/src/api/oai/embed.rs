use salvo::{
    oapi::{extract::JsonBody, ToParameters, ToResponse, ToSchema},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use crate::types::Array;
use std::path::Path;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};

trait EmbeddingModelExt {
    fn from_name(name: &str) -> Self;
}

impl EmbeddingModelExt for EmbeddingModel {
    fn from_name(name: &str) -> Self {
        match name {

            // Add other models as needed
            /*
            /// sentence-transformers/all-MiniLM-L6-v2
            AllMiniLML6V2,
            /// Quantized sentence-transformers/all-MiniLM-L6-v2
            AllMiniLML6V2Q,
            /// BAAI/bge-base-en-v1.5
            BGEBaseENV15,
            /// Quantized BAAI/bge-base-en-v1.5
            BGEBaseENV15Q,
            /// BAAI/bge-large-en-v1.5
            BGELargeENV15,
            /// Quantized BAAI/bge-large-en-v1.5
            BGELargeENV15Q,
            /// BAAI/bge-small-en-v1.5 - Default
            BGESmallENV15,
            /// Quantized BAAI/bge-small-en-v1.5
            BGESmallENV15Q,
            /// nomic-ai/nomic-embed-text-v1
            NomicEmbedTextV1,
            /// nomic-ai/nomic-embed-text-v1.5
            NomicEmbedTextV15,
            /// Quantized v1.5 nomic-ai/nomic-embed-text-v1.5
            NomicEmbedTextV15Q,
            /// sentence-transformers/paraphrase-MiniLM-L6-v2
            ParaphraseMLMiniLML12V2,
            /// Quantized sentence-transformers/paraphrase-MiniLM-L6-v2
            ParaphraseMLMiniLML12V2Q,
            /// sentence-transformers/paraphrase-mpnet-base-v2
            ParaphraseMLMpnetBaseV2,
            /// BAAI/bge-small-zh-v1.5
            BGESmallZHV15,
            /// intfloat/multilingual-e5-small
            MultilingualE5Small,
            /// intfloat/multilingual-e5-base
            MultilingualE5Base,
            /// intfloat/multilingual-e5-large
            MultilingualE5Large,
            /// mixedbread-ai/mxbai-embed-large-v1
            MxbaiEmbedLargeV1,
            /// Quantized mixedbread-ai/mxbai-embed-large-v1
            MxbaiEmbedLargeV1Q,
            */
            // 帮我把这些模型名字填进去

            "AllMiniLML6V2" => EmbeddingModel::AllMiniLML6V2,
            "AllMiniLML6V2Q" => EmbeddingModel::AllMiniLML6V2Q,
            "BGEBaseENV15" => EmbeddingModel::BGEBaseENV15,
            "BGEBaseENV15Q" => EmbeddingModel::BGEBaseENV15Q,
            "BGELargeENV15" => EmbeddingModel::BGELargeENV15,
            "BGELargeENV15Q" => EmbeddingModel::BGELargeENV15Q,
            "BGESmallENV15" => EmbeddingModel::BGESmallENV15,
            "BGESmallENV15Q" => EmbeddingModel::BGESmallENV15Q,
            "NomicEmbedTextV1" => EmbeddingModel::NomicEmbedTextV1,
            "NomicEmbedTextV15" => EmbeddingModel::NomicEmbedTextV15,
            "NomicEmbedTextV15Q" => EmbeddingModel::NomicEmbedTextV15Q,
            "ParaphraseMLMiniLML12V2" => EmbeddingModel::ParaphraseMLMiniLML12V2,
            "ParaphraseMLMiniLML12V2Q" => EmbeddingModel::ParaphraseMLMiniLML12V2,
            "ParaphraseMLMpnetBaseV2" => EmbeddingModel::ParaphraseMLMpnetBaseV2,
            "BGESmallZHV15" => EmbeddingModel::BGESmallZHV15,
            "MultilingualE5Small" => EmbeddingModel::MultilingualE5Small,
            "MultilingualE5Base" => EmbeddingModel::MultilingualE5Base,
            "MultilingualE5Large" => EmbeddingModel::MultilingualE5Large,
            "MxbaiEmbedLargeV1" => EmbeddingModel::MxbaiEmbedLargeV1,
            "MxbaiEmbedLargeV1Q" => EmbeddingModel::MxbaiEmbedLargeV1Q,



            _ => panic!("Unsupported model name"),
        }
    }
}

#[derive(Debug, Default, Clone, Deserialize, ToSchema, ToParameters)]
#[serde(default)]
pub struct EmbedRequest {
    input: Array<String>,
    mode_name: String,
}


#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbedData {
    object: String,
    index: usize,
    embeds: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize, ToSchema, ToResponse)]
pub struct EmbedResponse {
    object: String,
    model: String,
    data: Vec<EmbedData>,

}

/// Generate a embedding vector for the given text, with layer number specified for producing the embedding.
#[endpoint(responses((status_code = 200, body = EmbedResponse)))]
pub async fn embeds(
    _depot: &mut Depot,
    req: JsonBody<EmbedRequest>,
) -> Json<EmbedResponse> {

    let future = async move {
        let model_name = req.mode_name.clone();

        let model = TextEmbedding::try_new(InitOptions {
            model_name: EmbeddingModel::from_name(&model_name),
            show_download_progress: true,
            cache_dir: Path::new("assets\\models\\embed").to_path_buf(),
            ..Default::default()
        }).expect("Failed to initialize model");

        let embedding_result = model
            .embed(Vec::from(req.input.clone()), None)
            .expect("Failed to get embedding"); // Use expect to handle error
        print!("Embedding result: {:?}", embedding_result);

        embedding_result
    };

    let embedding_result = tokio::spawn(future).await.expect("spawn failed");

    Json(EmbedResponse {
        object: "embeds".into(),
        model: "multilingual-e5-small".into(),
        data: vec![EmbedData {
            object: "embed".into(),
            index: 0,
            embeds: embedding_result,
        }],
    })
}