from flask import Flask, request, render_template, redirect, url_for                                                                                                                                                                                                                                                                                                                            
import os, json, re
from datetime import datetime
import uuid
import whisper

app = Flask(__name__)

TEAM_FILE = 'data/team.json'
TASK_FILE = 'data/tasks.json'

def load_json(file_path,default):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(default, file, indent=2)
        return default
    with open(file_path, 'r') as file:
        return json.load(file)
    
def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


# transcribe audio
modal = whisper.load_model("medium").to("cuda")
def transcribe_audio(path):
    print("Transcribing:", path)
    print("Exists:", os.path.exists(path))
    result = modal.transcribe(path)
    print(result["text"])
    return result["text"]
    


# task detection
def extract_tasks_from_text(text):
    sentences = re.split(r'[.!?]', text)
    trigger_words = ['need to','should','must','fix','update','design','write','task', 'todo', 'reminder', 'action item', 'follow up']
    
    tasks = []
    for s in sentences:
        s_clean = s.strip().lower()
        if any(t in s_clean for t in trigger_words):
            tasks.append(s.strip())
    return tasks

# task assignment

def assign_tasks(task_list, team):
    assigned_output = []

    for task in task_list:
        assigned_to = None
        
        # Direct name matching
        for member in team:
            if member["name"].lower() in task.lower():
                assigned_to = member["name"]
                break
    
        # If no direct match found, skill-based assignment
        if assigned_to is None:
            best_match = None
            best_score = 0
            
            for member in team:  
                score = 0
                for skill in member["skills"]:
                    if skill.lower() in task.lower():
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_match = member["name"]
            
            assigned_to = best_match

        assigned_output.append({
            "id": str(uuid.uuid4()),
            "task": task,
            "assigned_to": assigned_to,
            "priority": "medium",
            "deadline": None,
            "dependencies": []
        })

    return assigned_output

# -----------------------------------------------------------------

@app.route("/")
def index():
    tasks = load_json(TASK_FILE,[])
    team = load_json(TEAM_FILE,[])
    return render_template("index.html", tasks=tasks, team=team)

@app.route("/upload", methods=["POST"])
def upload_audio():
    if 'audio' not in request.files:
        return redirect(url_for('index'))

    f = request.files['audio']

    # Ensure folders exist
    os.makedirs("uploads", exist_ok=True)

    # Safe path
    filename = f.filename.replace(" ", "_")
    path = os.path.join(os.getcwd(), "uploads", filename)

    # Save file
    f.save(path)

    # Debug print
    print("Saved audio at:", path)
    print("File exists?", os.path.exists(path))

    transcript = transcribe_audio(path)

    team = load_json(TEAM_FILE, [])
    task_extrected = extract_tasks_from_text(transcript)
    assigned = assign_tasks(task_extrected, team)

    existing = load_json(TASK_FILE, [])
    existing.extend(assigned)
    save_json(TASK_FILE, existing)

    return redirect(url_for('index'))


@app.route("/tasks/delete/<tid>")
def delete_task(tid):
    tasks = load_json(TASK_FILE, [])
    tasks = [t for t in tasks if t["id"] != tid]
    save_json(TASK_FILE, tasks)
    return redirect(url_for('index'))

@app.route("/tasks/clear")
def clear_tasks():
    save_json(TASK_FILE, [])
    return redirect(url_for('index'))

@app.route("/team/add", methods=["POST"])
def add_team():
    name = request.form["name"]
    role = request.form["role"]
    skills = request.form["skills"].split(",")

    team = load_json(TEAM_FILE, [])
    team.append({"name": name, "role": role, "skills": [s.strip() for s in skills]})
    save_json(TEAM_FILE, team)
    return redirect(url_for('index'))

@app.route("/team/delete/<name>")
def delete_team(name):
    team = load_json(TEAM_FILE, [])
    team = [m for m in team if m["name"].lower() != name.lower()]
    save_json(TEAM_FILE, team)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)