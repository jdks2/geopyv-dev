use thiserror::Error;

/// Library-wide error type. Variants are added per module as each is translated.
#[derive(Debug, Error)]
pub enum Error {
    /// Image file could not be opened or decoded.
    #[error("image load error: {0}")]
    ImageLoad(String),

    /// File path does not exist.
    #[error("file not found: {0}")]
    FileNotFound(String),

    /// Domain validation failure (value out of range, wrong shape, missing required input, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Mesh generation failure (CDT insertion or refinement error).
    #[error("mesh generation error: {0}")]
    MeshGeneration(String),

    /// File I/O or serialisation error.
    #[error("IO error: {0}")]
    Io(String),

    /// `.pyv` file header does not start with the expected magic bytes `b"GPYV"`.
    #[error("invalid file format: missing GPYV magic header")]
    InvalidMagic,

    /// `.pyv` file has a format version that this library cannot read.
    #[error("unsupported .pyv format version {0}; expected 1 or 2")]
    UnsupportedVersion(u8),

    /// Solve-dependent data was requested before `solve()` was called.
    #[error("{0} has not been solved; call solve() first")]
    NotSolved(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}
