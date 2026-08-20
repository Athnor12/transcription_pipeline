# Transcription Pipeline (Whisper & Gemini)

Une infrastructure auto-hébergée complète basée sur Docker pour l'automatisation de la transcription audio et de l'extraction de texte à partir d'images (OCR / Vision).

Le système s'intègre directement avec **Nextcloud** pour offrir un flux de travail fluide : 
déposez vos fichiers multimédias dans des dossiers synchronisés, et récupérez vos fichiers Markdown/Texte générés automatiquement.

---

## Architecture du Projet

transcription_pipeline/
├── nextcloud/            # Serveur Nextcloud (interface utilisateur & stockage)
├── whisper/              # Service de transcription audio local (Faster-Whisper / GPU)
│   └── app/
│       └── main.py       # Worker Python surveillant les dossiers audio
├── pipeline-gemini/      # Service d'OCR / Vision via l'API Google Gemini
│   └── app/
│       └── main.py       # Worker Python surveillant les dossiers d'images
├── docker-compose.yml    # Orchestration unifiée de la stack Docker
└── .env                  # Variables d'environnement & clés API
