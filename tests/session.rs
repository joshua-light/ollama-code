use std::fs;
use std::sync::{Mutex, MutexGuard};

use ollama_code::message::{Message, Role};
use ollama_code::session::Session;

/// Global mutex to serialize all session tests — they share the XDG_DATA_HOME env var.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// RAII guard that sets XDG_DATA_HOME for the duration of a test and restores it on drop.
/// Also holds the ENV_LOCK to serialize env var mutations.
struct SessionTestGuard {
    _lock: MutexGuard<'static, ()>,
    prev: Option<String>,
    _dir: tempfile::TempDir,
}

impl SessionTestGuard {
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let prev = std::env::var("XDG_DATA_HOME").ok();
        std::env::set_var("XDG_DATA_HOME", dir.path());
        Self {
            _lock: lock,
            prev,
            _dir: dir,
        }
    }

    fn data_dir(&self) -> &std::path::Path {
        self._dir.path()
    }
}

impl Drop for SessionTestGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
    }
}

#[test]
fn session_new_creates_files() {
    let _guard = SessionTestGuard::new();

    let session = Session::new().unwrap();
    let session_path = session.path().to_path_buf();

    assert!(session_path.join("messages.jsonl").exists());
    assert!(session_path.join("debug.log").exists());

    let debug_content = fs::read_to_string(session_path.join("debug.log")).unwrap();
    assert!(debug_content.contains("SESSION_START"));

    let id = session.id().to_string();
    assert!(id.len() > 15, "session id too short: {}", id);
    assert!(id.contains('_'));
}

#[test]
fn session_log_message() {
    let _guard = SessionTestGuard::new();

    let mut session = Session::new().unwrap();
    let path = session.path().to_path_buf();

    session.log_message(&Message::user("Hello!"));
    session.log_message(&Message::assistant("Hi there."));
    drop(session);

    let content = fs::read_to_string(path.join("messages.jsonl")).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2);

    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert!(first.get("ts").is_some());
    assert_eq!(first["role"].as_str(), Some("user"));
    assert_eq!(first["content"].as_str(), Some("Hello!"));

    let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(second["role"].as_str(), Some("assistant"));
}

#[test]
fn session_resume_loads_messages() {
    let _guard = SessionTestGuard::new();

    let mut session = Session::new().unwrap();
    let id = session.id().to_string();

    session.log_message(&Message::system("You are helpful."));
    session.log_message(&Message::user("What is 2+2?"));
    session.log_message(&Message::assistant("4"));
    drop(session);

    let (resumed, messages) = Session::resume(&id).unwrap();
    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[0].role, Role::System));
    assert!(matches!(messages[1].role, Role::User));
    assert!(matches!(messages[2].role, Role::Assistant));
    assert_eq!(messages[1].content, "What is 2+2?");
    assert_eq!(messages[2].content, "4");

    let debug_content = fs::read_to_string(resumed.path().join("debug.log")).unwrap();
    assert!(debug_content.contains("SESSION_RESUME"));
}

#[test]
fn session_resume_nonexistent_fails() {
    let _guard = SessionTestGuard::new();

    let result = Session::resume("nonexistent-id");
    assert!(result.is_err());
}

#[test]
fn session_list_recent() {
    let _guard = SessionTestGuard::new();

    drop(Session::new().unwrap());
    drop(Session::new().unwrap());

    let recent = Session::list_recent(10).unwrap();
    assert!(recent.len() >= 2);
    assert!(recent[0] >= recent[recent.len() - 1]);
}

#[test]
fn session_list_recent_limit() {
    let _guard = SessionTestGuard::new();

    for _ in 0..5 {
        drop(Session::new().unwrap());
    }
    let recent = Session::list_recent(3).unwrap();
    assert!(recent.len() <= 3);
}

#[test]
fn session_find_by_prefix() {
    let _guard = SessionTestGuard::new();

    let session = Session::new().unwrap();
    let id = session.id().to_string();
    drop(session);

    let prefix = &id[..8];
    let found = Session::find_by_prefix(prefix).unwrap();
    assert!(found.is_some());
    assert!(found.unwrap().starts_with(prefix));
}

#[test]
fn session_find_by_prefix_no_match() {
    let _guard = SessionTestGuard::new();

    drop(Session::new().unwrap());
    let found = Session::find_by_prefix("9999-99-99").unwrap();
    assert!(found.is_none());
}

#[test]
fn session_record_trim_and_resume() {
    let guard = SessionTestGuard::new();

    let mut session = Session::new().unwrap();
    let id = session.id().to_string();

    session.log_message(&Message::system("System"));
    session.log_message(&Message::user("Q1"));
    session.log_message(&Message::assistant("A1"));
    session.log_message(&Message::user("Q2"));
    session.log_message(&Message::assistant("A2"));

    session.record_trim(2);
    drop(session);

    // Verify meta.json
    let meta_path = guard
        .data_dir()
        .join("ollama-code")
        .join("sessions")
        .join(&id)
        .join("meta.json");
    assert!(meta_path.exists());
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
    assert_eq!(meta["trim_watermark"], 2);

    // Resume: should skip 2 messages after system
    let (_resumed, messages) = Session::resume(&id).unwrap();
    // Original: System, Q1, A1, Q2, A2 (5)
    // After trim watermark: System, Q2, A2 (3)
    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[0].role, Role::System));
    assert_eq!(messages[1].content, "Q2");
    assert_eq!(messages[2].content, "A2");
}

#[test]
fn session_drop_logs_end() {
    let _guard = SessionTestGuard::new();

    let session = Session::new().unwrap();
    let path = session.path().to_path_buf();
    drop(session);

    let debug = fs::read_to_string(path.join("debug.log")).unwrap();
    assert!(debug.contains("SESSION_END"));
}
