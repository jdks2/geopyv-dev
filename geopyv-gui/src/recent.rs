use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const MAX_RECENT: usize = 5;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RecentProjects {
    pub paths: Vec<PathBuf>,
}

impl RecentProjects {
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("geopyv").join("recent.json"))
    }

    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        let Ok(bytes) = std::fs::read(&path) else {
            return Self::default();
        };
        serde_json::from_slice(&bytes).unwrap_or_default()
    }

    pub fn save(&self) {
        let Some(path) = Self::config_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_vec_pretty(self) {
            let _ = std::fs::write(path, json);
        }
    }

    pub fn push(&mut self, p: &Path) {
        self.paths.retain(|x| x != p);
        self.paths.insert(0, p.to_path_buf());
        self.paths.truncate(MAX_RECENT);
        self.save();
    }

    pub fn remove(&mut self, p: &Path) {
        self.paths.retain(|x| x != p);
        self.save();
    }
}
