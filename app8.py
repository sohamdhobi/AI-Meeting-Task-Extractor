
import os
import json
import uuid
import re

from flask import Flask, request, render_template, redirect, url_for

import whisper
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

app = Flask(__name__)

TEAM_FILE = "data/team.json"
TASK_FILE = "data/tasks.json"

os.makedirs("data", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ----------------- MODELS -----------------


WHISPER_MODEL_NAME = "small"
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
if torch.cuda.is_available():
    whisper_model.to("cuda")

# spaCy
nlp = spacy.load("en_core_web_sm")

# sentence transformer (embeddings)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------- JSON HELPERS -----------------


def load_json(file_path, default):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


# ----------------- TRANSCRIPTION -----------------


def transcribe_audio(path):
    print("Transcribing:", path)
    result = whisper_model.transcribe(path)
    print("Transcript:", result["text"])
    return result["text"]


# ----------------- NLP UTILITIES -----------------

TRIGGER_VERBS = {
    "need",
    "should",
    "must",
    "fix",
    "update",
    "design",
    "write",
    "implement",
    "create",
    "add",
    "change",
    "improve",
    "optimize",
    "complete",
    "deploy",
    "test",
    "refactor",
    "tackle",
}

SKIP_PATTERNS = [
    r"didn't you",
    r"did you",
    r"you worked",
    r"you work",
    r"right\?",
    r"isn't it",
    r"aren't you",
    r"good with",
]


def split_sentences(text):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def is_question(sent_text):
    return sent_text.strip().endswith("?")


def is_conversational(sent_text):
    lower = sent_text.lower()
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, lower):
            return True
    return False


def is_dependency_only(sent_text):
    lower = sent_text.lower()
    dep_starts = ["this depends", "this is blocked", "this needs to wait"]
    return any(lower.startswith(start) for start in dep_starts)


def is_context_sentence(sent_text):
    lower = sent_text.lower()
    context_starts = ["this needs to be done", "this is", "it's", "this can wait"]
    return any(lower.startswith(start) for start in context_starts)


def has_task_verb(doc_sent):
    for tok in doc_sent:
        if tok.lemma_ in TRIGGER_VERBS:
            return True
    return False


def extract_clean_task_description(sent_text):
    """Extract and clean task description from sentence"""
    text = sent_text.strip()

    if "tackle this" in text.lower():
        return None

    text = re.sub(r"^[A-Z][a-z]+,\s*", "", text)

    # Extract after trigger phrases
    triggers = ["need someone to", "someone should", "need to", "we need", "should", "must"]
    for trigger in triggers:
        if trigger in text.lower():
            parts = re.split(trigger, text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                text = parts[1].strip()
                break

    # Remove context/deadline phrases
    text = re.sub(r"\s+(since|this needs|this is|it's|that users)\s+.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+before\s+\w+day's\s+release", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+for the next sprint", "", text, flags=re.IGNORECASE)

    # Remove leading filler words
    text = re.sub(r"^(and|oh,?\s+and|also)\s+", "", text, flags=re.IGNORECASE)

    text = text.strip().rstrip(".,!?")

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text


def is_valid_task(task_text):
    if not task_text or len(task_text) < 10:
        return False
    vague = ["tackle this", "do this", "handle this", "work on this"]
    if task_text.lower() in vague:
        return False
    return True


def extract_tasks_raw(full_text):
    
    sentences = split_sentences(full_text)
    tasks_raw = []
    current_person = None

    for i, sent in enumerate(sentences):
        if is_question(sent) or is_conversational(sent) or is_dependency_only(sent):
            continue

        # Check context sentence but allow "we should tackle"
        if is_context_sentence(sent):
            continue

        doc = nlp(sent)

        # Track person mentions
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if names:
            current_person = names[0]

        # Check if "you" refers to current person
        mentioned_in_context = []
        if current_person and "you" in sent.lower():
            mentioned_in_context.append(current_person)
        if names:
            mentioned_in_context.extend(names)

        if not has_task_verb(doc):
            continue

        # Special handling for "tackle this" - look back further for context
        cleaned = None
        if "tackle this" in sent.lower():
            for lookback in range(1, min(4, i + 1)):
                prev_sent = sentences[i - lookback]
                prev_lower = prev_sent.lower()

                if "database" in prev_lower and ("performance" in prev_lower or "slow" in prev_lower):
                    cleaned = "Optimize database performance"
                    # Look for person mention in sentences between problem and "tackle this"
                    for j in range(i - lookback, i + 1):
                        check_sent = sentences[j]
                        check_names = [
                            ent.text for ent in nlp(check_sent).ents if ent.label_ == "PERSON"
                        ]
                        if check_names:
                            mentioned_in_context = check_names
                    if not mentioned_in_context and current_person:
                        mentioned_in_context = [current_person]
                    break
                elif "performance" in prev_lower and "slow" in prev_lower:
                    cleaned = "Improve performance"
                    break

        if not cleaned:
            cleaned = extract_clean_task_description(sent)

        if not is_valid_task(cleaned):
            continue

        context_start = max(0, i - 2)
        context_end = min(len(sentences), i + 2)
        full_context = " ".join(sentences[context_start:context_end])

        tasks_raw.append(
            {
                "id": str(uuid.uuid4()),
                "sentence": sent,
                "task": cleaned,
                "mentioned_names": mentioned_in_context,
                "context": full_context,
                "original_sentence": sent,
                "sent_idx": i,
            }
        )

    return tasks_raw


def find_dependency_sentences(full_text):
    
    sentences = split_sentences(full_text)
    dep_phrases = ["depends on", "blocked by", "after", "once", "completed first", "then"]
    dep_sents = []
    for i, sent in enumerate(sentences):
        if any(phrase in sent.lower() for phrase in dep_phrases):
            dep_sents.append({"text": sent, "sent_idx": i})
    print("DEPENDENCY SENTENCES FOUND:", [d["text"] for d in dep_sents])
    return dep_sents


def build_team_profiles(team):
    if not team:
        return [], np.zeros((0, 384))
    profiles = []
    for member in team:
        skills = " ".join(member.get("skills", []))
        profile = f"{member['name']} {member['role']} {skills}"
        profiles.append(profile)
    embeddings = embedding_model.encode(profiles)
    return profiles, embeddings


def assign_single_task(task_text, original_sentence, mentioned_names, team, team_embeddings):
    
    mentioned_lower = [n.lower() for n in mentioned_names]
    for member in team:
        if member["name"].lower() in mentioned_lower:
            return member["name"]

    # ---------- SKILL MATCH PRIORITY ----------
    task_lower = task_text.lower()
    skill_matches = []

    for member in team:
        for skill in member.get("skills", []):
            if skill.lower() in task_lower:
                skill_matches.append(member["name"])
                break

    if skill_matches:
        return skill_matches[0]  # assign first matching member

    # ---------- SEMANTIC MATCH BACKUP ----------
    if not team or team_embeddings.shape[0] == 0:
        return None

    task_emb = embedding_model.encode([original_sentence])[0].reshape(1, -1)
    sims = cosine_similarity(task_emb, team_embeddings).flatten()
    best_idx = int(np.argmax(sims))

    if sims[best_idx] > 0.25:
        return team[best_idx]["name"]

    return None


def detect_deadline(context_text):
    t = context_text.lower()
    patterns = [
        (r"by\s+tomorrow\s+evening", "Tomorrow evening"),
        (r"tomorrow\s+evening", "Tomorrow evening"),
        (r"by\s+end\s+of\s+(this\s+)?week", "End of this week"),
        (r"end\s+of\s+(this\s+)?week", "End of this week"),
        (r"before\s+friday'?s?\s+release", "Friday"),
        (r"before\s+friday", "Friday"),
        (r"by\s+friday", "Friday"),
        (r"wait\s+until\s+next\s+monday", "Next Monday"),
        (r"until\s+next\s+monday", "Next Monday"),
        (r"next\s+monday", "Next Monday"),
        (r"plan.*?for\s+wednesday", "Wednesday"),
        (r"for\s+wednesday", "Wednesday"),
        (r"by\s+wednesday", "Wednesday"),
        # days
        (r"by\s+monday", "Monday"),
        (r"by\s+tuesday", "Tuesday"),
        (r"by\s+wednesday", "Wednesday"),
        (r"by\s+thursday", "Thursday"),
        (r"by\s+friday", "Friday"),
        (r"by\s+saturday", "Saturday"),
        (r"by\s+sunday", "Sunday"),
    ]

    for pattern, formatted in patterns:
        if re.search(pattern, t):
            return formatted
    return None


def detect_priority(context_text):
    t = context_text.lower()
    if any(w in t for w in ["critical", "blocker", "blocking", "high priority", "urgent", "asap"]):
        return "High"
    if any(w in t for w in ["low priority", "nice to have", "can wait"]):
        return "Low"
    return "Medium"


def deduplicate_tasks(tasks, sim_threshold=0.85):
    
    if not tasks:
        return tasks

    texts = [t["task"] for t in tasks]
    embs = embedding_model.encode(texts)

    keep_indices = []
    for i in range(len(tasks)):
        is_dup = False
        for j in keep_indices:
            sim = cosine_similarity(embs[i].reshape(1, -1), embs[j].reshape(1, -1))[0][0]
            if sim >= sim_threshold:
                is_dup = True
                break
        if not is_dup:
            keep_indices.append(i)

    # Keep in original order
    kept = [tasks[i] for i in keep_indices]
    return kept


def link_dependencies(tasks, dependency_sentences, sim_threshold=0.30):
    
    if not tasks or not dependency_sentences:
        return tasks

    # Build embeddings for tasks
    task_texts = [t["task"] for t in tasks]
    task_embs = embedding_model.encode(task_texts)

    for dep in dependency_sentences:
        dep_text = dep["text"]
        dep_idx = dep["sent_idx"]

        # find child: last task with sent_idx < dep_idx
        child_candidates = [(i, t["sent_idx"]) for i, t in enumerate(tasks) if t.get("sent_idx", -1) < dep_idx]
        child_candidates.sort(key=lambda x: x[1], reverse=True)  # latest first

        if child_candidates:
            child_idx = child_candidates[0][0]
        else:
            # fallback: use the second-best similarity as child (if available)
            dep_emb = embedding_model.encode([dep_text])[0].reshape(1, -1)
            sims_full = cosine_similarity(dep_emb, task_embs).flatten()
            sorted_idxs = np.argsort(sims_full)[::-1]
            if len(sorted_idxs) >= 2:
                child_idx = int(sorted_idxs[1])
            elif len(sorted_idxs) == 1:
                child_idx = int(sorted_idxs[0])
            else:
                continue

        # find parent via similarity
        dep_emb = embedding_model.encode([dep_text])[0].reshape(1, -1)
        sims = cosine_similarity(dep_emb, task_embs).flatten()
        parent_idx = int(np.argmax(sims))
        if sims[parent_idx] < sim_threshold:
            # if top similarity is below threshold, skip linking
            continue

        # If parent == child, try next best parent if possible
        if parent_idx == child_idx:
            sorted_idxs = np.argsort(sims)[::-1]
            for idx in sorted_idxs:
                if int(idx) != child_idx and sims[int(idx)] >= sim_threshold:
                    parent_idx = int(idx)
                    break
            else:
                # couldn't find a distinct parent above threshold
                continue

        parent = tasks[parent_idx]
        child = tasks[child_idx]

        child.setdefault("dependencies", [])
        if parent["task"] not in child["dependencies"]:
            child["dependencies"].append(parent["task"])

            print(f"Linked dependency: child='{child['task']}' <- parent='{parent['task']}'")

    return tasks


def build_tasks(full_text, team):
    
    tasks_raw = extract_tasks_raw(full_text)
    _, team_embeddings = build_team_profiles(team)

    # Build tasks preserving transcript order and sent_idx
    tasks = []
    for raw in tasks_raw:
        assignee = assign_single_task(
            raw["task"], raw["original_sentence"], raw["mentioned_names"], team, team_embeddings
        )
        deadline = detect_deadline(raw["context"])
        priority = detect_priority(raw["context"])

        tasks.append(
            {
                "id": raw["id"],
                "task": raw["task"],
                "assigned_to": assignee,
                "priority": priority,
                "deadline": deadline,
                "dependencies": [],
                "sent_idx": raw["sent_idx"],  # preserve for dependency linking
            }
        )

    # Deduplicate BEFORE linking dependencies to avoid double tasks being linked badly.
    tasks = deduplicate_tasks(tasks, sim_threshold=0.85)

    # Extract dependency sentences (with indices)
    dep_sentences = find_dependency_sentences(full_text)

    # Link dependencies using positional+semantic hybrid algorithm
    tasks = link_dependencies(tasks, dep_sentences, sim_threshold=0.30)

    # Remove helper field 'sent_idx' before returning/persisting
    for t in tasks:
        if "sent_idx" in t:
            t.pop("sent_idx", None)

    return tasks


# ----------------- FLASK ROUTES -----------------


@app.route("/")
def index():
    tasks = load_json(TASK_FILE, [])
    team = load_json(TEAM_FILE, [])
    return render_template("index.html", tasks=tasks, team=team)


@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio" not in request.files:
        return redirect(url_for("index"))

    f = request.files["audio"]
    filename = f.filename.replace(" ", "_")
    path = os.path.join("uploads", filename)
    f.save(path)

    transcript = transcribe_audio(path)
    team = load_json(TEAM_FILE, [])
    new_tasks = build_tasks(transcript, team)

    print("\n=== EXTRACTED TASKS ===")
    print(json.dumps(new_tasks, indent=2))

    existing = load_json(TASK_FILE, [])
    existing.extend(new_tasks)
    save_json(TASK_FILE, existing)

    try:
        os.remove(path)
    except Exception:
        pass

    return redirect(url_for("index"))


@app.route("/tasks/delete/<tid>")
def delete_task(tid):
    tasks = load_json(TASK_FILE, [])
    tasks = [t for t in tasks if t["id"] != tid]
    save_json(TASK_FILE, tasks)
    return redirect(url_for("index"))


@app.route("/tasks/clear")
def clear_tasks():
    save_json(TASK_FILE, [])
    return redirect(url_for("index"))


@app.route("/team/add", methods=["POST"])
def add_team():
    name = request.form["name"]
    role = request.form["role"]
    skills = request.form.get("skills", "")
    skills_list = [s.strip() for s in skills.split(",") if s.strip()]

    team = load_json(TEAM_FILE, [])
    team.append({"name": name, "role": role, "skills": skills_list})
    save_json(TEAM_FILE, team)
    return redirect(url_for("index"))


@app.route("/team/delete/<name>")
def delete_team(name):
    team = load_json(TEAM_FILE, [])
    team = [m for m in team if m["name"].lower() != name.lower()]
    save_json(TEAM_FILE, team)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
