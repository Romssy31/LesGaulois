# Stem Reveal : un son construit couche par couche, en vidéo

Le script génère une vidéo verticale 1080×1920 à 30 i/s, d'au moins 62 s, avec les textes en anglais :

| Étape | Titre | Ce qu'on entend |
|---|---|---|
| 0/5 | MUFFLED VOCALS | voix seule, filtrée (sans aigus) |
| 1/5 | VOCALS | voix claire |
| 2/5 | + MELODY | + instruments |
| 3/5 | + BASS | + basse |
| 4/5 | + DRUMS | + batterie, filtre qui monte sur les 2 dernières mesures, puis un demi-temps de silence |
| 5/5 | FULL SONG | **le drop** : le refrain repart avec le son original complet, et un spectre circulaire réagit à la musique |

**Tout est calé sur la musique :**
- le tempo, les temps et les débuts de mesure sont détectés automatiquement ;
- chaque changement tombe exactement sur un début de mesure ;
- la comète finit son tour pile au moment où la couche suivante entre ;
- la lueur, les bulles et le cercle central pulsent sur la grosse caisse ;
- le refrain (ou le drop) est trouvé automatiquement : c'est la partie la plus forte avec de la voix.

## Utilisation

```bash
python3 stem_reveal.py son.mp3 -o video.mp4 --handle @tonpseudo --credit "Titre — Artiste"
```

Options utiles :
- `--start 61.5` : forcer le début du refrain (en secondes) si le choix automatique ne convient pas
- `--stems dossier/` : utiliser tes propres pistes (`vocals`, `other`, `bass`, `drums` en .wav ou .mp3) au lieu de la séparation automatique
- `--min-duration 62` : durée minimale (TikTok : plus de 60 s)
- `--accent "#9FB4FF"` : couleur d'accent (comète, lueur, appel à l'abonnement)
- `--cta "FOLLOW FOR THE NEXT DROP" --cta2 "new songs every day"` : textes de fin

Le script écrit aussi `video.plan.json`, avec le BPM, la position du refrain et l'heure de début de chaque étape.

## Installation

```bash
pip install -r requirements.txt     # torch, demucs, librosa, skia-python…
# ffmpeg doit être installé ; la police Inter est attendue dans /usr/share/fonts/opentype/inter
```

Le premier lancement télécharge le modèle Demucs `htdemucs` (environ 80 Mo, depuis dl.fbaipublicfiles.com).
Sur un processeur 4 cœurs, la séparation prend environ 1 min pour un son de 3 min, et le rendu environ 2 min.

## Droits d'auteur

Si tu utilises un son du commerce, TikTok peut couper le son de la vidéo ou la bloquer. Le plus sûr :
un son de la bibliothèque TikTok, de la musique libre de droits, ou tes propres productions.
