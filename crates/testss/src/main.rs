
use text_splitter::{ChunkConfig, TextSplitter};
// Can also use anything else that implements the ChunkSizer
// trait from the text_splitter crate.
use tokenizers::Tokenizer;
use std::env;

fn main() {
    env::set_var("HF_ENDPOINT", "https://hf-mirror.com");
    env::set_var("HF_HOME", "./assets/models/hf_hub");
    let tokenizer = Tokenizer::from_pretrained("intfloat/multilingual-e5-small", None).unwrap();
    let max_tokens = 1000;
    let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));

    let chunks = splitter.chunks("your document text");
//将chunks打印出来
    for chunk in chunks {
        println!("{}", chunk);
    }
}
