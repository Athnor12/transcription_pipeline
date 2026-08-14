import os
import glob
import shutil
import time
from faster_whisper import WhisperModel

# --- CONFIGURATION VIA VARIABLES D'ENVIRONNEMENT ---
DOSSIER_SOURCE = os.getenv("DOSSIER_SOURCE", "/data/chapitres_audio")
DOSSIER_TEXTES = os.getenv("DOSSIER_TEXTES", "/data/chapitres_ecrits")
DOSSIER_ARCHIVES = os.getenv("DOSSIER_ARCHIVES", "/data/archives_audio")
MODEL_SIZE = os.getenv("MODEL_SIZE", "turbo") # ou "medium", "small"
DEVICE = os.getenv("DEVICE", "cpu") # "cuda" si tu as un GPU NVIDIA, "cpu" sinon

def traiter_audios():
    for d in [DOSSIER_TEXTES, DOSSIER_ARCHIVES]:
        os.makedirs(d, exist_ok=True)

    extensions = ('*.mp3', '*.m4a', '*.wav', '*.flac')
    fichiers_audio = []
    for ext in extensions:
        fichiers_audio.extend(glob.glob(os.path.join(DOSSIER_SOURCE, ext)))

    if not fichiers_audio:
        return

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Chargement du modèle Whisper ({MODEL_SIZE} sur {DEVICE})...")
    # compute_type="int8" permet de tourner ultra-vite sur CPU sans perdre en qualité
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")

    for chemin_audio in fichiers_audio:
        nom_fichier = os.path.basename(chemin_audio)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing : {nom_fichier}")

        # Transcription avec suppression des hallucinations de silence
        segments, info = model.transcribe(
            chemin_audio,
            beam_size=5,
            no_speech_threshold=0.6,
            condition_on_previous_text=False
        )

        # Reconstitution du texte
        texte_complet = ""
        for segment in segments:
            texte_complet += segment.text + " "

        # Nettoyage automatique des dialogues
        texte_propre = texte_complet.replace(" tiret", "\n—").replace(" Tiret", "\n—")

        # Sauvegarde TXT
        nom_txt = os.path.splitext(nom_fichier)[0] + ".txt"
        chemin_txt = os.path.join(DOSSIER_TEXTES, nom_txt)
        with open(chemin_txt, "w", encoding="utf-8") as f:
            f.write(texte_propre.strip())

        # Archivage Audio
        shutil.move(chemin_audio, os.path.join(DOSSIER_ARCHIVES, nom_fichier))
        print(f"✅ Traité : {nom_txt}")

if __name__ == "__main__":
    print("Démarrage du service de transcription automatique...")
    while True:
        try:
            traiter_audios()
        except Exception as e:
            print(f"Erreur durant le traitement : {e}")
        # Vérifie le dossier toutes les 30 secondes
        time.sleep(30)
