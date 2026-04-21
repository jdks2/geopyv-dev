use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ProjectError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// The folder exists but has no `project.json`.
    NotAProject(PathBuf),
    /// `project.json` exists but has an unrecognised version field.
    UnsupportedVersion(u32),
}

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "project.json parse error: {e}"),
            Self::NotAProject(p) => write!(
                f,
                "{} does not contain a project.json — not a geopyv project",
                p.display()
            ),
            Self::UnsupportedVersion(v) => {
                write!(f, "project.json version {v} is not supported by this build")
            }
        }
    }
}

impl From<std::io::Error> for ProjectError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ProjectError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

// ---------------------------------------------------------------------------
// project.json schema
// ---------------------------------------------------------------------------

const CURRENT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMeta {
    pub name: String,
    pub version: u32,
}

// ---------------------------------------------------------------------------
// Subdirectory layout
// ---------------------------------------------------------------------------

/// All subdirectory names relative to the project root.
const SUBDIRS: &[&str] = &[
    "Images/Data",
    "Images/Calibration",
    "Templates",
    "Subsets",
    "Meshes",
    "Sequences",
    "Particles",
    "Fields",
    "Out",
];

// ---------------------------------------------------------------------------
// Project
// ---------------------------------------------------------------------------

/// Runtime representation of an open project.
#[derive(Debug, Clone)]
pub struct Project {
    pub root: PathBuf,
    pub meta: ProjectMeta,
}

impl Project {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Initialise a brand-new project in `root` (folder must already exist).
    /// Creates all subdirectories and writes `project.json`.
    pub fn create(root: &Path) -> Result<Self, ProjectError> {
        let name = root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "project".to_string());

        ensure_subdirs(root)?;

        let meta = ProjectMeta {
            name,
            version: CURRENT_VERSION,
        };
        write_meta(root, &meta)?;

        Ok(Self {
            root: root.to_path_buf(),
            meta,
        })
    }

    /// Open an existing project rooted at `root`.
    /// Validates `project.json`; ensures any missing subdirectories are created.
    pub fn open(root: &Path) -> Result<Self, ProjectError> {
        let manifest = root.join("project.json");
        if !manifest.exists() {
            return Err(ProjectError::NotAProject(root.to_path_buf()));
        }

        let bytes = std::fs::read(&manifest)?;
        let meta: ProjectMeta = serde_json::from_slice(&bytes)?;

        if meta.version != CURRENT_VERSION {
            return Err(ProjectError::UnsupportedVersion(meta.version));
        }

        // Silently repair any subdirectories that have gone missing.
        ensure_subdirs(root)?;

        Ok(Self {
            root: root.to_path_buf(),
            meta,
        })
    }

    // -----------------------------------------------------------------------
    // Typed path accessors
    // -----------------------------------------------------------------------

    pub fn images_data_dir(&self) -> PathBuf {
        self.root.join("Images/Data")
    }

    pub fn images_calibration_dir(&self) -> PathBuf {
        self.root.join("Images/Calibration")
    }

    pub fn templates_dir(&self) -> PathBuf {
        self.root.join("Templates")
    }

    pub fn subsets_dir(&self) -> PathBuf {
        self.root.join("Subsets")
    }

    pub fn meshes_dir(&self) -> PathBuf {
        self.root.join("Meshes")
    }

    pub fn sequences_dir(&self) -> PathBuf {
        self.root.join("Sequences")
    }

    pub fn particles_dir(&self) -> PathBuf {
        self.root.join("Particles")
    }

    pub fn fields_dir(&self) -> PathBuf {
        self.root.join("Fields")
    }

    pub fn out_dir(&self) -> PathBuf {
        self.root.join("Out")
    }

    // -----------------------------------------------------------------------
    // File listing helpers
    // -----------------------------------------------------------------------

    /// Sorted lexicographic list of image files in `/Images/Data/`.
    pub fn list_images(&self) -> Vec<PathBuf> {
        list_files_with_exts(&self.images_data_dir(), &["jpg", "jpeg", "png", "tif", "tiff"])
    }

    /// Sorted list of template `.json` files in `/Templates/`.
    pub fn list_templates(&self) -> Vec<PathBuf> {
        list_files_with_exts(&self.templates_dir(), &["json"])
    }

    /// Sorted list of `.pyv` files in `/Subsets/`.
    pub fn list_subsets(&self) -> Vec<PathBuf> {
        list_files_with_exts(&self.subsets_dir(), &["pyv"])
    }

    /// Sorted list of standalone `.pyv` files directly in `/Meshes/`, plus
    /// the names of any sequence-linked subfolders.
    pub fn list_meshes(&self) -> MeshEntries {
        let dir = self.meshes_dir();
        let mut standalone = Vec::new();
        let mut sequence_folders = Vec::new();

        if let Ok(rd) = std::fs::read_dir(&dir) {
            let mut entries: Vec<_> = rd.flatten().collect();
            entries.sort_by_key(|e| e.file_name());

            for entry in entries {
                let path = entry.path();
                if path.is_dir() {
                    sequence_folders.push(path);
                } else if matches_ext(&path, &["pyv"]) {
                    standalone.push(path);
                }
            }
        }

        MeshEntries {
            standalone,
            sequence_folders,
        }
    }

    /// Sorted list of `.pyv` files in `/Sequences/`.
    pub fn list_sequences(&self) -> Vec<PathBuf> {
        list_files_with_exts(&self.sequences_dir(), &["pyv"])
    }

    /// Sorted list of `.pyv` files in `/Particles/`.
    pub fn list_particles(&self) -> Vec<PathBuf> {
        list_files_with_exts(&self.particles_dir(), &["pyv"])
    }

    /// Sorted list of `.pyv` files in `/Fields/`.
    pub fn list_fields(&self) -> Vec<PathBuf> {
        list_files_with_exts(&self.fields_dir(), &["pyv"])
    }

    // -----------------------------------------------------------------------
    // Path helpers used by solve sessions
    // -----------------------------------------------------------------------

    /// Returns the path for a named `.pyv` file in `subdir`, e.g.
    /// `project.pyv_path("Subsets", "my_subset")` → `.../Subsets/my_subset.pyv`.
    pub fn pyv_path(&self, subdir: &str, name: &str) -> PathBuf {
        self.root.join(subdir).join(format!("{name}.pyv"))
    }

    /// Returns the path for a per-frame mesh file inside a sequence subfolder.
    pub fn frame_pyv_path(&self, sequence_name: &str, frame: usize) -> PathBuf {
        self.root
            .join("Meshes")
            .join(sequence_name)
            .join(format!("frame_{frame:03}.pyv"))
    }

    /// Creates the sequence mesh subfolder if it does not exist.
    pub fn ensure_sequence_mesh_dir(&self, sequence_name: &str) -> Result<PathBuf, ProjectError> {
        let dir = self.root.join("Meshes").join(sequence_name);
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Persist any in-memory changes to `project.json`.
    pub fn save_meta(&self) -> Result<(), ProjectError> {
        write_meta(&self.root, &self.meta)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn ensure_subdirs(root: &Path) -> Result<(), std::io::Error> {
    for sub in SUBDIRS {
        std::fs::create_dir_all(root.join(sub))?;
    }
    Ok(())
}

fn write_meta(root: &Path, meta: &ProjectMeta) -> Result<(), ProjectError> {
    let json = serde_json::to_vec_pretty(meta)?;
    std::fs::write(root.join("project.json"), json)?;
    Ok(())
}

fn matches_ext(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| exts.iter().any(|x| x.eq_ignore_ascii_case(e)))
        .unwrap_or(false)
}

/// Returns a sorted list of files in `dir` whose extensions match `exts`.
fn list_files_with_exts(dir: &Path, exts: &[&str]) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && matches_ext(p, exts))
        .collect();

    files.sort();
    files
}

// ---------------------------------------------------------------------------
// Mesh entry type
// ---------------------------------------------------------------------------

/// Result of `Project::list_meshes`.
pub struct MeshEntries {
    /// Standalone single-pair `.pyv` files directly in `/Meshes/`.
    pub standalone: Vec<PathBuf>,
    /// Subfolders whose names correspond to sequences (each contains frame files).
    pub sequence_folders: Vec<PathBuf>,
}
