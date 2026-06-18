// Calque orchestrator — Axum web front door.
//
// Responsibilities:
//   * serve the static web UI from ./static-site
//   * expose /health for the deploy script + tunnel
//   * proxy /api/* to the local Python worker (image effects + GPU generation)
//
// Mirrors the Almanach deployment pattern (single binary + ServeDir + systemd).

use std::time::Duration;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{uri::Uri, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{any, get},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

#[derive(Clone)]
struct AppState {
    /// Base URL of the Python worker, e.g. http://127.0.0.1:8001
    worker_url: String,
    http: reqwest::Client,
}

async fn health() -> &'static str {
    "ok"
}

/// Proxy everything under /api/* to the worker, stripping the /api prefix.
async fn proxy(State(state): State<AppState>, req: Request) -> Result<Response, StatusCode> {
    let path = req.uri().path();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    // /api/effect/watercolor -> /effect/watercolor
    let forward_path = path.strip_prefix("/api").unwrap_or(path);
    let target = format!("{}{}{}", state.worker_url, forward_path, query);

    let _uri: Uri = target.parse().map_err(|_| StatusCode::BAD_GATEWAY)?;

    let method = req.method().clone();
    let headers = req.headers().clone();
    let body_bytes = axum::body::to_bytes(req.into_body(), usize::MAX)
        .await
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let resp = state
        .http
        .request(method, &target)
        .headers(headers)
        .body(body_bytes)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("worker proxy error: {e}");
            StatusCode::BAD_GATEWAY
        })?;

    let status = resp.status();
    let mut builder = Response::builder().status(status);
    for (k, v) in resp.headers() {
        builder = builder.header(k, v);
    }
    let bytes = resp.bytes().await.map_err(|_| StatusCode::BAD_GATEWAY)?;
    builder
        .body(Body::from(bytes))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
        .map(IntoResponse::into_response)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3002);
    let worker_url =
        std::env::var("WORKER_URL").unwrap_or_else(|_| "http://127.0.0.1:8001".into());
    let static_dir = std::env::var("STATIC_DIR").unwrap_or_else(|_| "./static-site".into());

    let state = AppState {
        worker_url: worker_url.clone(),
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(300)) // generation can be slow
            .build()?,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/*rest", any(proxy))
        .fallback_service(ServeDir::new(&static_dir))
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("🎨 Calque orchestrator on http://{addr}  (worker: {worker_url})");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
