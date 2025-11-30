# AI Meeting Task Extractor

An intelligent Flask-based web application that automatically transcribes meeting audio recordings and extracts actionable tasks using NLP and machine learning. The system intelligently assigns tasks to team members based on skill matching and context, detects priorities and deadlines, and identifies task dependencies.

## Features

### Core Capabilities
- **Audio Transcription**: Converts meeting recordings to text using OpenAI's Whisper model
- **Intelligent Task Extraction**: Uses NLP to identify actionable items from natural conversation
- **Smart Assignment**: Automatically assigns tasks to team members based on:
  - Direct mentions in the conversation
  - Skill matching with task requirements
  - Semantic similarity between tasks and team profiles
- **Priority Detection**: Automatically identifies task priorities (High/Medium/Low) from context
- **Deadline Recognition**: Extracts deadlines from natural language (e.g., "by Friday", "end of week")
- **Dependency Linking**: Detects and links task dependencies using hybrid positional and semantic analysis
- **Duplicate Detection**: Removes redundant tasks using semantic similarity
- **Team Management**: Add and manage team members with roles and skills

### Technical Features
- Sentence-level analysis with spaCy
- Semantic embeddings using Sentence Transformers
- GPU acceleration support for faster transcription
- Clean and intuitive web interface
- Persistent JSON storage for tasks and team data

## Architecture

### Key Components

**NLP Pipeline:**
- **Whisper** (small model): Audio transcription
- **spaCy** (en_core_web_sm): Sentence parsing and entity recognition
- **SentenceTransformer** (all-MiniLM-L6-v2): Semantic embeddings for matching and similarity

**Task Processing:**
1. Sentence segmentation and filtering
2. Trigger verb detection (need, should, must, fix, etc.)
3. Context extraction and cleaning
4. Person mention tracking
5. Task deduplication
6. Dependency linking
7. Assignment and metadata enrichment

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster transcription

### Setup Instructions

1. **Clone or download the project**
```bash
cd AI-Meeting-Task-Extractor

```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirement.txt
```

4. **Download spaCy language model**
```bash
python -m spacy download en_core_web_sm
```

5. **Create required directories** (auto-created on first run)
```bash
mkdir data uploads
```

## Usage

### Starting the Application

1. **Run the Flask server**
```bash
python app.py
```

2. **Access the web interface**
   - Open your browser and navigate to: `http://127.0.0.1:5000`

### Adding Team Members

Before uploading meetings, add your team members:

1. Go to the **Team Management** section
2. Enter:
   - **Name**: Team member's name
   - **Role**: Their job title (e.g., Backend Developer, Designer)
   - **Skills**: Comma-separated list (e.g., Python, API, Database)
3. Click **Add Team Member**

**Example:**
- Name: Alex
- Role: Backend Developer
- Skills: Python, Database, API

### Processing Meeting Audio

1. **Record or prepare your meeting audio** (supported formats: WAV, MP3, M4A, etc.)
2. Click **Choose File** and select your audio file
3. Click **Upload & Process**
4. Wait for transcription and task extraction (may take 30-60 seconds depending on audio length)
5. View extracted tasks with automatic assignments, priorities, and deadlines

### Managing Tasks

- **View Tasks**: All extracted tasks appear in a table with details
- **Delete Task**: Click the delete button next to any task
- **Clear All**: Remove all tasks at once

## How It Works

### Task Extraction Logic

The system identifies tasks by:
1. **Filtering out**: Questions, conversational phrases, dependency-only statements
2. **Looking for trigger verbs**: need, should, must, fix, implement, create, etc.
3. **Tracking context**: Maintains awareness of who is being discussed
4. **Cleaning descriptions**: Removes filler words and extracts core actions

### Assignment Algorithm

**Priority Order:**
1. **Direct mentions**: If someone is explicitly mentioned with the task
2. **Skill matching**: Matches task keywords with team member skills
3. **Semantic similarity**: Uses AI embeddings to find best fit based on role and expertise

### Dependency Detection

Dependencies are identified through:
- Phrases like "depends on", "blocked by", "after", "once completed"
- Positional analysis (tasks mentioned before dependency statements)
- Semantic similarity to link related tasks
- Hybrid scoring to avoid false positives

### Priority & Deadline Detection

**Priorities:**
- **High**: critical, blocker, urgent, asap
- **Low**: nice to have, can wait
- **Medium**: default

**Deadlines:** Extracted from phrases like:
- "by tomorrow evening"
- "end of this week"
- "before Friday's release"
- Specific days (Monday, Tuesday, etc.)

## Project Structure

```
meeting-task-extractor/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── data/
│   ├── team.json         # Team member data
│   └── tasks.json        # Extracted tasks
├── uploads/              # Temporary audio files
└── README.md             # This file
```

## Configuration

### Model Selection

You can change models in `app.py`:

```python
# Whisper model options: tiny, base, small, medium, large
WHISPER_MODEL_NAME = "small"  # Balance of speed and accuracy

# For better accuracy (slower):
WHISPER_MODEL_NAME = "medium"

# For faster processing (less accurate):
WHISPER_MODEL_NAME = "base"
```

### Similarity Thresholds

Adjust sensitivity in the code:
- **Task deduplication**: `sim_threshold=0.85` (higher = stricter)
- **Dependency linking**: `sim_threshold=0.30` (lower = more permissive)
- **Assignment matching**: `sims[best_idx] > 0.25` (minimum confidence)

## Example Workflow

**Sample Meeting Transcript:**
> "Hey team, we need to optimize the database performance. Alex, can you tackle this? Also, Sarah should update the API documentation before Friday's release. This depends on the database work being completed first."

**Extracted Tasks:**
1. **Task**: Optimize database performance
   - **Assigned to**: Alex
   - **Priority**: Medium
   - **Deadline**: None
   
2. **Task**: Update the API documentation
   - **Assigned to**: Sarah
   - **Priority**: Medium
   - **Deadline**: Friday
   - **Dependencies**: Optimize database performance

## Troubleshooting

### Common Issues

**"No module named 'whisper'"**
```bash
pip install openai-whisper
```

**"Can't find model 'en_core_web_sm'"**
```bash
python -m spacy download en_core_web_sm
```

**Slow transcription**
- Use a smaller Whisper model (`tiny` or `base`)
- GPU acceleration will significantly speed up processing

**Tasks not being extracted**
- Ensure clear action words are used (need, should, must, fix, etc.)
- Check that team members are added before processing
- Verify audio quality is sufficient for accurate transcription

**Wrong assignments**
- Add more specific skills to team members
- Use explicit names in meeting discussions
- Review and adjust similarity thresholds

## Performance Tips

1. **GPU Acceleration**: If you have a CUDA GPU, Whisper will automatically use it
2. **Audio Quality**: Better audio = better transcription = better task extraction
3. **Clear Communication**: Explicit action items work best ("Alex needs to fix the bug")
4. **Team Profiles**: More detailed skills improve assignment accuracy

## Dependencies

- **Flask**: Web framework
- **openai-whisper**: Audio transcription
- **spacy**: NLP and sentence parsing
- **sentence-transformers**: Semantic embeddings
- **scikit-learn**: Similarity calculations
- **numpy**: Numerical operations
- **torch**: Deep learning framework

## License

This project is provided as-is for educational and commercial use.

## Contributing

Feel free to enhance the system by:
- Adding support for more audio formats
- Improving task extraction patterns
- Enhancing the UI
- Adding export functionality (CSV, JSON, etc.)
- Integrating with project management tools

## Future Enhancements

- Real-time transcription during meetings
- Multi-language support
- Integration with Slack, Jira, Asana
- Calendar integration for deadline tracking
- Email notifications for task assignments
- Task status tracking and completion
- Meeting summary generation

---

**Made with ❤️ using AI and NLP**
