# Training Document Processor

Automated Azure pipeline that converts MP4 training videos into structured
step-by-step training manuals (.docx) with screenshots.- v3

## How it works

1. Drop an MP4 into the `videos` container in Azure Blob Storage
2. Azure Function triggers automatically
3. Video Indexer transcribes the video and extracts keyframes
4. GPT-4o Vision captions each screenshot
5. GPT-4o builds the document structure and writes training content
6. A `.docx` manual is saved to the `outputs` container
7. You receive an email with a download link
8. Original video is deleted automatically

## Project structure

```
training-doc-processor/
├── function_app.py       ← Main orchestrator + blob trigger + watchdog
├── video_indexer.py      ← Azure Video Indexer API client
├── openai_client.py      ← GPT-4o vision + text calls
├── document_builder.py   ← python-docx document assembly
├── notifier.py           ← Azure Communication Services email
├── requirements.txt      ← Python dependencies
├── host.json             ← Azure Function App config
└── local.settings.json   ← Local dev credentials (never commit)
```

## Azure services required

| Service | Name |
|---|---|
| Blob Storage | trainingdocstore |
| Video Indexer | training-doc-ai |
| Azure OpenAI | yav12 (gpt-4o) |
| Function App | training-doc-processor |
| Communication Services | (see setup below) |

## Blob Storage containers

| Container | Purpose |
|---|---|
| `videos` | Drop MP4 files here to trigger pipeline |
| `outputs` | Generated .docx files appear here |
| `keyframes` | Temporary keyframe storage |
| `intermediate` | JSON checkpoints (enables resume on failure) |

## Environment variables

Set these in Azure Function App → Configuration → Application Settings:

```
STORAGE_CONNECTION_STRING   = <your storage connection string>
STORAGE_ACCOUNT_NAME        = trainingdocstore
VIDEO_INDEXER_ACCOUNT_ID    = 3276c0f1-684c-4d40-87ad-c93e26e80f50
VIDEO_INDEXER_LOCATION      = eastus
SUBSCRIPTION_ID             = dc88a7df-a0e8-46a0-8617-59c39a25c376
RESOURCE_GROUP              = training-doc-generator
VI_ACCOUNT_NAME             = training-doc-ai
AZURE_OPENAI_ENDPOINT       = https://yav12.openai.azure.com/
AZURE_OPENAI_KEY            = <your key>
AZURE_OPENAI_DEPLOYMENT     = gpt-4o
NOTIFICATION_EMAIL_FROM     = <verified sender email>
NOTIFICATION_EMAIL_TO       = <recipient email>
ACS_CONNECTION_STRING       = <ACS connection string>
```

## GitHub deployment setup

1. Create a new GitHub repository named `training-doc-processor`
2. Push this code to the `main` branch
3. In Azure Portal:
   - Go to `training-doc-processor` Function App
   - Deployment Center → Source: GitHub
   - Select your repo and `main` branch
   - Save → Azure auto-deploys on every push

## Setting up Azure Communication Services (email)

1. Portal → Create resource → "Communication Services"
2. Name: `training-doc-comms` → Resource Group: `training-doc-generator`
3. Once created → Email → Domains → Add a free Azure domain
4. Copy the connection string → add to app settings as `ACS_CONNECTION_STRING`
5. Set `NOTIFICATION_EMAIL_FROM` to `donotreply@<your-azure-domain>`
6. Set `NOTIFICATION_EMAIL_TO` to your email address

## Resume on failure

The pipeline saves JSON checkpoints after each expensive step:
- `intermediate/<video_name>/transcript.json`
- `intermediate/<video_name>/keyframes.json`
- `intermediate/<video_name>/captions.json`
- `intermediate/<video_name>/structure.json`
- `intermediate/<video_name>/content.json`

If the pipeline fails mid-way, simply re-upload the video — it will
resume from the last successful checkpoint automatically.

## Cost estimate (per 40-min video)

| Service | Cost |
|---|---|
| Video Indexer | ~$2.40 |
| GPT-4o Vision (keyframes) | ~$0.60 |
| GPT-4o Text (structure + writing) | ~$1.80 |
| Blob Storage | ~$0.02 |
| Azure Functions | ~$0.05 |
| **Total** | **~$4.87** |
