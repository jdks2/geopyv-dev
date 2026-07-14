//! Serialisation / deserialisation for geopyv-dev result objects.
//!
//! Translates `geopyv/src/geopyv/io.py` (save/load functions).
//!
//! # Format
//!
//! Every `.pyv` file written by this module has the following layout:
//!
//! ```text
//! [0..4]   magic  b"GPYV"
//! [4]      version  0x01 (uncompressed, legacy) or 0x02 (zstd-compressed)
//! [5..]    bincode v2 (standard config) encoded GeopyvObject,
//!          optionally compressed with zstd (level 3) for version 0x02
//! ```
//!
//! Version 0x01 files (written by earlier releases) are still readable.
//! New files are always written with version 0x02 (zstd-compressed).
//!
//! The type tag is embedded in the bincode stream via serde's enum encoding.
//!
//! # Breaking change from geopyv Python
//!
//! The Python `io.py` uses `pickle` with a `dict`-based data model. The Rust
//! format is intentionally incompatible: old `.pyv` files cannot be read here,
//! and files written here cannot be read by the Python package. Users who need
//! files from an older version should use the corresponding Python release.

use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    field::FieldSolution,
    mesh::MeshSolution,
    particle::ParticleSolution,
    sequence::SequenceSolution,
    speckle::Speckle,
    subset::SubsetSolution,
    Error,
};

// ---------------------------------------------------------------------------
// Magic header
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 4] = b"GPYV";
/// Legacy uncompressed format; still readable on load.
const VERSION_UNCOMPRESSED: u8 = 0x01;
/// Current format: zstd-compressed bincode payload.
const VERSION_ZSTD: u8 = 0x02;

// ---------------------------------------------------------------------------
// Tagged union for all serialisable object types
// ---------------------------------------------------------------------------

/// Discriminated union of all geopyv-dev result types that can be saved to a
/// `.pyv` file.  The variant tag is embedded in the bincode stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeopyvObject {
    Subset(SubsetSolution),
    Mesh(MeshSolution),
    Field(FieldSolution),
    Sequence(SequenceSolution),
    Particle(ParticleSolution),
    Speckle(Speckle),
}

// ---------------------------------------------------------------------------
// save / load
// ---------------------------------------------------------------------------

/// Serialise a [`GeopyvObject`] to a `.pyv` file at `path`.
///
/// The file is created (or truncated if it already exists).
/// Format: 4-byte magic `b"GPYV"` + 1-byte version `0x02` + zstd-compressed
/// bincode payload.
pub fn save<P: AsRef<Path>>(path: P, object: &GeopyvObject) -> Result<(), Error> {
    let encoded =
        bincode::serde::encode_to_vec(object, bincode::config::standard())
            .map_err(|e| Error::Io(e.to_string()))?;
    let compressed = zstd::encode_all(encoded.as_slice(), 3)
        .map_err(|e| Error::Io(e.to_string()))?;
    let mut file = std::fs::File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&[VERSION_ZSTD])?;
    file.write_all(&compressed)?;
    Ok(())
}

/// Load a [`GeopyvObject`] from a `.pyv` file at `path`.
///
/// Returns [`Error::InvalidMagic`] if the file does not start with `b"GPYV"`.
/// Returns [`Error::UnsupportedVersion`] for any version byte other than
/// `0x01` (legacy uncompressed) or `0x02` (zstd-compressed).
pub fn load<P: AsRef<Path>>(path: P) -> Result<GeopyvObject, Error> {
    let mut file = std::fs::File::open(path.as_ref()).map_err(|e| {
        Error::FileNotFound(format!("{}: {}", path.as_ref().display(), e))
    })?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(Error::InvalidMagic);
    }

    let mut ver_buf = [0u8; 1];
    file.read_exact(&mut ver_buf)?;

    let mut raw = Vec::new();
    file.read_to_end(&mut raw)?;

    let data = match ver_buf[0] {
        VERSION_UNCOMPRESSED => raw,
        VERSION_ZSTD => zstd::decode_all(raw.as_slice())
            .map_err(|e| Error::Io(e.to_string()))?,
        v => return Err(Error::UnsupportedVersion(v)),
    };

    let (object, _) =
        bincode::serde::decode_from_slice::<GeopyvObject, _>(
            &data,
            bincode::config::standard(),
        )
        .map_err(|e| Error::Io(e.to_string()))?;

    Ok(object)
}

// ---------------------------------------------------------------------------
// Unit tests (Phase 2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use ndarray::{array, Array1, Array2};

    use std::sync::Arc;
    use crate::{
        mesh::{MeshSolution, NodeResult},
        sequence::SequenceSolution,
    };

    /// Build a minimal MeshSolution with known values for round-trip tests.
    fn make_mesh_solution() -> MeshSolution {
        let nodes = array![[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let elements = array![[0usize, 1, 2]];
        let centroids = crate::mesh::compute_centroids(&nodes, &elements);
        MeshSolution {
            nodes,
            elements,
            boundary: vec![0, 1, 2],
            exclusions: vec![],
            centroids,
            areas: array![0.5_f64],
            warps: array![[0.1_f64, 0.2, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            displacements: array![[0.1_f64, 0.2], [0.3, 0.4], [0.5, 0.6]],
            c_zncc: array![0.999_f64, 0.998, 0.997],
            p: array![[0.1_f64, 0.2, 0.0, 0.0, 0.0, 0.0],
                      [0.3, 0.4, 0.0, 0.0, 0.0, 0.0],
                      [0.5, 0.6, 0.0, 0.0, 0.0, 0.0]],
            seed_node: 0,
            mesh_order: 1,
            subset_order: 1,
            iterations: ndarray::array![0u32, 0, 0],
            norms: ndarray::array![0.0_f64, 0.0, 0.0],
            f_img_path: PathBuf::new(),
            g_img_path: PathBuf::new(),
        }
    }

    /// Round-trip MeshSolution through save/load.
    #[test]
    fn test_mesh_round_trip() {
        let original = make_mesh_solution();
        let tmp = std::env::temp_dir().join("geopyv_test_mesh.pyv");
        save(&tmp, &GeopyvObject::Mesh(original.clone())).unwrap();

        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert_eq!(m.nodes, original.nodes);
                assert_eq!(m.elements, original.elements);
                assert_eq!(m.boundary, original.boundary);
                assert!((m.areas[0] - original.areas[0]).abs() < 1e-15);
                assert_eq!(m.seed_node, original.seed_node);
                assert_eq!(m.mesh_order, original.mesh_order);
            }
            _ => panic!("expected Mesh variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Round-trip SequenceSolution through save/load.
    #[test]
    fn test_sequence_round_trip() {
        let sol = SequenceSolution {
            mesh_solutions: vec![Arc::new(make_mesh_solution()), Arc::new(make_mesh_solution())],
            mesh_paths: vec![],
            solved: true,
            unsolvable: false,
            override_log: vec![],
            reference_updates: vec![false, false],
            mesh_order: 1,
            first_f_img_path: None,
            boundary_region: crate::sequence::default_boundary_region(),
            exclusion_regions: vec![],
        };
        let tmp = std::env::temp_dir().join("geopyv_test_seq.pyv");
        save(&tmp, &GeopyvObject::Sequence(sol.clone())).unwrap();

        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Sequence(s) => {
                assert_eq!(s.mesh_solutions.len(), 2);
                assert!(s.solved);
                assert!(!s.unsolvable);
            }
            _ => panic!("expected Sequence variant"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// f64 values round-trip with bit-exact precision (Tier A).
    #[test]
    fn test_f64_precision() {
        let val = std::f64::consts::PI;
        let mut sol = make_mesh_solution();
        sol.displacements[[0, 0]] = val;
        let tmp = std::env::temp_dir().join("geopyv_test_f64.pyv");
        save(&tmp, &GeopyvObject::Mesh(sol)).unwrap();
        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert_eq!(m.displacements[[0, 0]], val);
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Invalid magic bytes return `Error::InvalidMagic`.
    #[test]
    fn test_invalid_magic() {
        let tmp = std::env::temp_dir().join("geopyv_test_bad_magic.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(b"XXXX\x01\x00\x00\x00").unwrap();
        }
        let result = load(&tmp);
        assert!(matches!(result, Err(Error::InvalidMagic)));
        let _ = std::fs::remove_file(tmp);
    }

    /// Wrong version byte returns `Error::UnsupportedVersion`.
    #[test]
    fn test_unsupported_version() {
        let tmp = std::env::temp_dir().join("geopyv_test_bad_ver.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(b"GPYV\x03\x00\x00\x00").unwrap();
        }
        let result = load(&tmp);
        assert!(matches!(result, Err(Error::UnsupportedVersion(0x03))));
        let _ = std::fs::remove_file(tmp);
    }

    /// A version 0x01 (legacy uncompressed) file is still readable.
    #[test]
    fn test_v1_backwards_compat() {
        let sol = make_mesh_solution();
        let encoded = bincode::serde::encode_to_vec(
            &GeopyvObject::Mesh(sol.clone()),
            bincode::config::standard(),
        )
        .unwrap();
        let tmp = std::env::temp_dir().join("geopyv_test_v1_compat.pyv");
        {
            let mut f = std::fs::File::create(&tmp).unwrap();
            f.write_all(b"GPYV\x01").unwrap();
            f.write_all(&encoded).unwrap();
        }
        let loaded = load(&tmp).unwrap();
        match loaded {
            GeopyvObject::Mesh(m) => {
                assert_eq!(m.nodes, sol.nodes);
            }
            _ => panic!("expected Mesh"),
        }
        let _ = std::fs::remove_file(tmp);
    }

    /// Loading a nonexistent file returns `Error::FileNotFound`.
    #[test]
    fn test_file_not_found() {
        let result = load("/nonexistent/path/file.pyv");
        assert!(matches!(result, Err(Error::FileNotFound(_))));
    }
}
