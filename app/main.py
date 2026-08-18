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
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")

def charger_modele():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Chargement du modèle Whisper ({MODEL_SIZE} sur {DEVICE})...", flush=True)
    # Chargement global une seule fois
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")


def traiter_audios(model):
    for d in [DOSSIER_TEXTES, DOSSIER_ARCHIVES]:
        os.makedirs(d, exist_ok=True)

    # Récupérer TOUS les fichiers du dossier sans se soucier des majuscules/minuscules
    extensions_valides = ('.mp3', '.m4a', '.wav', '.flac', '.aac', '.ogg')
    fichiers_audio = []
    
    if os.path.exists(DOSSIER_SOURCE):
        for f in os.listdir(DOSSIER_SOURCE):
            chemin_complet = os.path.join(DOSSIER_SOURCE, f)
            if os.path.isfile(chemin_complet) and f.lower().endswith(extensions_valides):
                fichiers_audio.append(chemin_complet)

    if not fichiers_audio:
        return

    for chemin_audio in fichiers_audio:
        nom_fichier = os.path.basename(chemin_audio)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing : {nom_fichier}", flush=True)

        # Transcription
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
        print(f"✅ Traité : {nom_txt}", flush=True)

if __name__ == "__main__":
    print("Démarrage du service de transcription automatique...", flush=True)
    
    # 1. Charger le modèle au démarrage (téléchargement immédiat)
    whisper_model = charger_modele()
    print("🚀 Modèle prêt ! Surveillance du dossier en cours...", flush=True)
    
    # 2. Boucle de traitement
    while True:
        try:
            traiter_audios(whisper_model)
        except Exception as e:
            print(f"Erreur durant le traitement : {e}", flush=True)
        time.sleep(10)
