use clap::error::ErrorKind::InvalidValue;
use clap::{ArgGroup, Error, Parser};
use inference_benchmarker::{run, RunConfiguration, TokenizeOptions, DistributionMode, BenchmarkKind, TokenizerSource};
use log::{debug, error};
use reqwest::Url;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::broadcast;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None, group(ArgGroup::new("group_profile").multiple(true)),group(ArgGroup::new("group_manual").multiple(true).conflicts_with("group_profile"))
)]
struct Args {
    /// The name of the tokenizer to use
    #[clap(short, long, env)]
    tokenizer_name: String,

    /// The source of the tokenizer to use (hub or local)
    #[clap(default_value = "hub", short, long, env, value_parser = parse_tokenizer_source)]
    tokenizer_source: TokenizerSource,

    /// The name of the model to use. If not provided, the same name as the tokenizer will be used.
    #[clap(long, env)]
    model_name: Option<String>,

    /// The maximum number of virtual users to use
    #[clap(default_value = "128", short, long, env, group = "group_manual")]
    max_vus: u64,
    /// The duration of each benchmark step
    #[clap(default_value = "120s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    duration: Duration,
    /// A list of rates of requests to send per second (only valid for the ConstantArrivalRate benchmark).
    #[clap(short, long, env)]
    rates: Option<Vec<f64>>,
    /// The number of rates to sweep through (only valid for the "sweep" benchmark)
    /// The rates will be linearly spaced up to the detected maximum rate
    #[clap(default_value = "10", long, env)]
    num_rates: u64,
    /// A benchmark profile to use
    #[clap(long, env, group = "group_profile")]
    profile: Option<String>,
    /// The kind of benchmark to run (throughput, sweep, optimum)
    #[clap(default_value = "sweep", short, long, env, group = "group_manual")]
    benchmark_kind: BenchmarkKind,
    /// The duration of the prewarm step ran before the benchmark to warm up the backend (JIT, caches, etc.)
    #[clap(default_value = "30s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    warmup: Duration,
    /// The URL of the backend to benchmark. Must be compatible with OpenAI Message API
    #[clap(default_value = "http://localhost:8000", short, long, env)]
    url: Url,

    /// The api key send to the [`url`] as Header "Authorization: Bearer {API_KEY}".
    #[clap(default_value = "", short, long, env)]
    api_key: String,

    /// Disable console UI
    #[clap(short, long, env)]
    no_console: bool,
    /// Constraints for prompt length.
    ///
    /// Format: comma-separated key=value pairs.
    ///
    /// Keys:
    /// * `num_tokens` — target number of prompt tokens
    /// * `min_tokens` — minimum number of prompt tokens
    /// * `max_tokens` — maximum number of prompt tokens
    /// * `dist` — distribution specification
    ///
    /// The `dist` value defines how token counts are sampled:
    /// - Normal distribution: `dist=normal:variance=5`
    /// - Log-normal distribution: `dist=log_normal:log_mean=6.2,log_std=1.3`
    /// - Gamma distribution: `dist=gamma:shape=0.7,scale=242`
    ///
    /// Examples:
    /// ```
    /// --prompt-options "num_tokens=200,min_tokens=190,max_tokens=210"
    /// --prompt-options "num_tokens=200,min_tokens=180,max_tokens=220,dist=normal:variance=10"
    /// --prompt-options "dist=log_normal:log_mean=6.2,log_std=1.3"
    /// ```
    #[clap(
        long,
        env,
        value_parser(parse_tokenizer_options),
        group = "group_manual"
    )]
    prompt_options: Option<TokenizeOptions>,

    /// Constraints for the generated text.
    ///
    /// Format: comma-separated key=value pairs.
    ///
    /// Keys:
    /// * `num_tokens` — target number of generated tokens
    /// * `min_tokens` — minimum number of generated tokens
    /// * `max_tokens` — maximum number of generated tokens
    /// * `dist` — distribution specification
    ///
    /// The `dist` value defines how generation lengths are sampled:
    /// - Normal distribution: `dist=normal:variance=5`
    /// - Log-normal distribution: `dist=log_normal:log_mean=6.2,log_std=1.3`
    ///
    /// Examples:
    /// ```
    /// --decode-options "num_tokens=200,min_tokens=190,max_tokens=210"
    /// --decode-options "num_tokens=200,min_tokens=180,max_tokens=220,dist=normal:variance=10"
    /// --decode-options "dist=log_normal:log_mean=6.5,log_std=1.1"
    /// ```
    #[clap(
        long,
        env,
        value_parser(parse_tokenizer_options),
        group = "group_manual"
    )]
    decode_options: Option<TokenizeOptions>,
    /// Hugging Face dataset to use for prompt generation
    #[clap(
        default_value = "hlarcher/inference-benchmarker",
        long,
        env,
        group = "group_manual"
    )]
    dataset: String,
    /// File to use in the Dataset
    #[clap(
        default_value = "share_gpt_filtered_small.json",
        long,
        env,
        group = "group_manual"
    )]
    dataset_file: String,
    /// Extra metadata to include in the benchmark results file, comma-separated key-value pairs.
    /// It can be, for example, used to include information about the configuration of the
    /// benched server.
    /// Example: --extra-meta "key1=value1,key2=value2"
    #[clap(long, env, value_parser(parse_key_val))]
    extra_meta: Option<HashMap<String, String>>,
    // A run identifier to use for the benchmark. This is used to identify the benchmark in the
    // results file.
    #[clap(long, env)]
    run_id: Option<String>,
}

fn parse_duration(s: &str) -> Result<Duration, Error> {
    humantime::parse_duration(s).map_err(|_| Error::new(InvalidValue))
}

fn parse_key_val(s: &str) -> Result<HashMap<String, String>, Error> {
    let mut key_val_map = HashMap::new();
    let items = s.split(",").collect::<Vec<&str>>();
    for item in items.iter() {
        let key_value = item.split("=").collect::<Vec<&str>>();
        if key_value.len() % 2 != 0 {
            return Err(Error::new(InvalidValue));
        }
        for i in 0..key_value.len() / 2 {
            key_val_map.insert(
                key_value[i * 2].to_string(),
                key_value[i * 2 + 1].to_string(),
            );
        }
    }
    Ok(key_val_map)
}

fn parse_tokenizer_options(s: &str) -> Result<TokenizeOptions, Error> {
    let mut opts = TokenizeOptions::new();
    let mut items = s.split(',').map(str::trim).peekable();

    while let Some(item) = items.next() {
        if item.starts_with("dist=") {
            // Collect all tokens belonging to the distribution
            let mut dist_str = String::from(item);
            while let Some(&next) = items.peek() {
                if next.starts_with("variance=")
                    || next.starts_with("log_mean=")
                    || next.starts_with("log_std=")
                    || next.starts_with("shape=")
                    || next.starts_with("scale=")
                {
                    dist_str.push(',');
                    dist_str.push_str(next);
                    items.next();
                } else {
                    break;
                }
            }
            opts.distribution_mode = parse_distribution_mode(&dist_str)?;
            continue;
        }

        // Regular key=value pairs
        let (key, val) = item.split_once('=').ok_or_else(|| Error::new(InvalidValue))?;
        match key {
            "num_tokens" => opts.num_tokens = Some(val.parse().map_err(|_| Error::new(InvalidValue))?),
            "min_tokens" => opts.min_tokens = val.parse().map_err(|_| Error::new(InvalidValue))?,
            "max_tokens" => opts.max_tokens = val.parse().map_err(|_| Error::new(InvalidValue))?,
            _ => return Err(Error::new(InvalidValue)),
        }
    }

    if opts.min_tokens > opts.max_tokens {
        return Err(Error::new(InvalidValue));
    }

    Ok(opts)
}


fn parse_distribution_mode(spec: &str) -> Result<DistributionMode, Error> {
    // Examples:
    // "dist=normal:variance=5"
    // "dist=log_normal:log_mean=6.2,log_std=1.3"
    // "dist=normal"

    let parts: Vec<&str> = spec.splitn(2, ':').collect();
    let dist_name = parts[0]
        .strip_prefix("dist=")
        .ok_or_else(|| Error::new(InvalidValue))?
        .to_lowercase();

    match dist_name.as_str() {
        "normal" => {
            if parts.len() == 1 {
                return Ok(DistributionMode::Normal { variance: 0 });
            }
            let mut variance = 0;
            for kv in parts[1].split(',') {
                let pair: Vec<&str> = kv.split('=').collect();
                if pair.len() != 2 || pair[0].trim() != "variance" {
                    return Err(Error::new(InvalidValue));
                }
                variance = pair[1].trim().parse::<u64>().map_err(|_| Error::new(InvalidValue))?;
            }
            Ok(DistributionMode::Normal { variance })
        }
        "log_normal" => {
            if parts.len() == 1 {
                return Err(Error::new(InvalidValue)); // needs params
            }
            let mut log_mean = None;
            let mut log_std = None;

            for kv in parts[1].split(',') {
                let pair: Vec<&str> = kv.split('=').collect();
                if pair.len() != 2 {
                    return Err(Error::new(InvalidValue));
                }
                match pair[0].trim() {
                    "log_mean" => {
                        log_mean = Some(pair[1].trim().parse::<f64>().map_err(|_| Error::new(InvalidValue))?);
                    }
                    "log_std" => {
                        log_std = Some(pair[1].trim().parse::<f64>().map_err(|_| Error::new(InvalidValue))?);
                    }
                    _ => return Err(Error::new(InvalidValue)),
                }
            }

            Ok(DistributionMode::LogNormal {
                log_mean: log_mean.ok_or_else(|| Error::new(InvalidValue))?,
                log_std: log_std.ok_or_else(|| Error::new(InvalidValue))?,
            })
        }
        "gamma" => {
            if parts.len() == 1 {
                return Err(Error::new(InvalidValue)); // needs params
            }
            let mut shape = None;
            let mut scale = None;

            for kv in parts[1].split(',') {
                let pair: Vec<&str> = kv.split('=').collect();
                if pair.len() != 2 {
                    return Err(Error::new(InvalidValue));
                }
                match pair[0].trim() {
                    "shape" => {
                        shape = Some(pair[1].trim().parse::<f64>().map_err(|_| Error::new(InvalidValue))?);
                    }
                    "scale" => {
                        scale = Some(pair[1].trim().parse::<f64>().map_err(|_| Error::new(InvalidValue))?);
                    }
                    _ => return Err(Error::new(InvalidValue)),
                }
            }

            Ok(DistributionMode::Gamma {
                shape: shape.ok_or_else(|| Error::new(InvalidValue))?,
                scale: scale.ok_or_else(|| Error::new(InvalidValue))?,
            })
        }
        _ => Err(Error::new(InvalidValue)),
    }
}

fn parse_tokenizer_source(s: &str) -> Result<TokenizerSource, Error> {
    match s.to_lowercase().as_str() {
        "hub" => Ok(TokenizerSource::Hub),
        "local" => Ok(TokenizerSource::Local),
        _ => Err(Error::new(InvalidValue)),
    }
}


#[tokio::main]
async fn main() {
    let args = Args::parse();
    let git_sha = option_env!("VERGEN_GIT_SHA").unwrap_or("unknown");
    println!(
        "Text Generation Inference Benchmark {} ({})",
        env!("CARGO_PKG_VERSION"),
        git_sha
    );

    let (stop_sender, _) = broadcast::channel(1);
    // handle ctrl-c
    let stop_sender_clone = stop_sender.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for ctrl-c");
        debug!("Received stop signal, stopping benchmark");
        stop_sender_clone
            .send(())
            .expect("Failed to send stop signal");
    });

    let stop_sender_clone = stop_sender.clone();
    // get HF token
    let token_env_key = "HF_TOKEN".to_string();
    let cache = hf_hub::Cache::from_env();
    let hf_token = match std::env::var(token_env_key).ok() {
        Some(token) => Some(token),
        None => cache.token(),
    };
    let model_name = args
        .model_name
        .clone()
        .unwrap_or(args.tokenizer_name.clone());
    let run_id = args
        .run_id
        .unwrap_or(uuid::Uuid::new_v4().to_string()[..7].to_string());
    let run_config = RunConfiguration {
        url: args.url,
        api_key: args.api_key,
        profile: args.profile.clone(),
        tokenizer_name: args.tokenizer_name.clone(),
        tokenizer_source: args.tokenizer_source.clone(),
        max_vus: args.max_vus,
        duration: args.duration,
        rates: args.rates,
        num_rates: args.num_rates,
        benchmark_kind: args.benchmark_kind.clone(),
        warmup_duration: args.warmup,
        interactive: !args.no_console,
        prompt_options: args.prompt_options.clone(),
        decode_options: args.decode_options.clone(),
        dataset: args.dataset.clone(),
        dataset_file: args.dataset_file.clone(),
        extra_metadata: args.extra_meta.clone(),
        hf_token,
        model_name,
        run_id,
    };
    let main_thread = tokio::spawn(async move {
        match run(run_config, stop_sender_clone).await {
            Ok(_) => {}
            Err(e) => {
                error!("Fatal: {:?}", e);
                println!("Fatal: {:?}", e)
            }
        };
    });
    let _ = main_thread.await;
}

#[test]
fn test_parse_tokenizer_options() {
    let expected = TokenizeOptions {
        num_tokens: Some(2500),
        min_tokens: 1,
        max_tokens: 131000,
        distribution_mode: DistributionMode::LogNormal {
            log_mean: 6.62,
            log_std: 1.4,
        }
    };
    let result = parse_tokenizer_options("num_tokens=2500,min_tokens=1,max_tokens=131000,dist=log_normal:log_mean=6.62,log_std=1.4");
    assert!(result.is_ok());
    assert_eq!(expected, result.unwrap());
}

#[test]
fn test_parse_tokenizer_source() {
    let result = parse_tokenizer_source("Hub");
    assert!(result.is_ok());
    assert_eq!(TokenizerSource::Hub, result.unwrap());
    let result = parse_tokenizer_source("hub");
    assert!(result.is_ok());
    assert_eq!(TokenizerSource::Hub, result.unwrap());
    let result = parse_tokenizer_source("local");
    assert!(result.is_ok());
    assert_eq!(TokenizerSource::Local, result.unwrap());
    let result = parse_tokenizer_source("random");
    assert!(result.is_err());
}