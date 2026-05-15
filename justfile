default: check

# Mirror of CI gates. Run before declaring any change done.
check:
    cargo fmt --all -- --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo clippy --all-targets --no-default-features -- -D warnings
    cargo clippy --all-targets --no-default-features --features ollama -- -D warnings
    cargo clippy --all-targets --no-default-features --features static -- -D warnings
    cargo clippy --all-targets --no-default-features --features rig-hook -- -D warnings
    cargo clippy --all-targets --no-default-features --features pricing -- -D warnings
    cargo test --all-features
    cargo build --examples --all-features

fmt:
    cargo fmt --all

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

test:
    cargo test --all-features

doc:
    cargo doc --all-features --no-deps
