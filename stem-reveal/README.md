# Stem Reveal : un son construit couche par couche, en vidéo

Un logiciel qui transforme n'importe quel son en vidéo TikTok verticale (1080×1920, au moins 62 s, textes en anglais).
Une bille part du centre, rebondit sur chaque temps et sort d'un anneau à chaque nouvelle couche :

| Étape | Titre | Ce qu'on entend |
|---|---|---|
| 0/5 | MUFFLED VOCALS | voix seule, filtrée (sans aigus) |
| 1/5 | VOCALS | voix claire |
| 2/5 | + MELODY | + instruments |
| 3/5 | + BASS | + basse |
| 4/5 | + DRUMS | + batterie, filtre qui monte sur les 2 dernières mesures, puis un demi-temps de silence |
| 5/5 | FULL SONG | **le drop** : le refrain repart avec le son original complet, et un spectre circulaire réagit à la musique |

## Deux types de vidéo

- **Couches** (ci-dessus) : 5 anneaux, chaque sortie de la bille ajoute une couche du son.
- **Échappée** : des centaines d'anneaux fins autour d'un petit cercle, avec un compteur au centre.
  Tant que la bille rebondit contre la paroi, la musique est **étouffée**. Quand elle passe par la fente qui tourne,
  les anneaux cassent un par un, le compteur descend et la musique devient **claire**.
  Les passages dans la fente commencent et finissent sur un temps (à la fin de chaque phrase de 4 mesures,
  de plus en plus longs), le refrain est entièrement clair et le compteur arrive à 0 pile à la fin, puis le spectre apparaît.
  Réglages : nombre d'anneaux, mesures avant le refrain, mesures de refrain (auto), durées min et max.

## Lancer le logiciel

1. Installe **Python 3.12** depuis https://www.python.org/downloads/ (sur Windows, coche « Add Python to PATH »).
2. Double-clique sur :
   - **Windows** : `Lancer Stem Reveal (Windows).bat`
   - **Mac** : `Lancer Stem Reveal (Mac).command` (la première fois : clic droit → Ouvrir)
3. Le premier lancement installe tout (5 à 10 minutes, une seule fois). Ensuite le logiciel s'ouvre dans ton navigateur.

## Comment ça marche

1. **Importer** : glisse ton son (MP3, WAV, M4A, FLAC, ou même une vidéo).
   Le logiciel sépare les pistes (voix, mélodie, basse, batterie) avec l'IA Demucs, puis détecte le tempo, chaque temps, chaque début de mesure et le refrain.
2. **Timing** : la forme d'onde affiche les mesures, la voix et le refrain choisi.
   - Clique ailleurs sur la forme d'onde, ou utilise ◀ / ▶, pour changer le refrain. Il se cale toujours sur une mesure.
   - La frise montre l'heure exacte de chaque étape, et tu peux **écouter le montage audio** avant de lancer le rendu.
   - **Durées** : choisis un préréglage (Courte ≈ 30 s, Standard ≈ 1 min, Longue ≈ 1 min 30), ou règle toi-même la longueur
     de chaque étape **en mesures** (le logiciel affiche l'équivalent en secondes). Tu peux aussi fixer une durée minimale et
     une durée maximale : la partie FULL SONG s'allonge ou se raccourcit, mesure par mesure, pour les respecter.
3. **Générer** : choisis ton pseudo, le titre, la couleur et les textes, puis clique sur « Générer la vidéo ». Tu la télécharges à la fin.

Les vidéos sont enregistrées dans `projects/videos/`.

**Tout est calé sur la musique :**
- chaque changement d'étape tombe exactement sur un début de mesure ;
- la bille rebondit sur chaque temps ;
- chaque anneau a une fente qui tourne. Sa vitesse est calculée pour que la bille passe dans la fente pile au début de mesure où la couche suivante entre, et jamais avant ;
- quand la bille sort d'un anneau, il éclate et disparaît, et la bille rebondit librement dans le suivant ;
- la lueur, les bulles et le cercle central pulsent sur la grosse caisse.

## Temps de calcul

Sur un ordinateur portable récent : environ 1 min pour séparer un son de 3 min (beaucoup moins avec une carte graphique NVIDIA),
puis environ 2 min de rendu. Un son déjà importé une fois est mis en cache : il se recharge instantanément.

## En ligne de commande (optionnel)

```bash
python3 stem_reveal.py son.mp3 -o video.mp4 --handle @tonpseudo --credit "Titre — Artiste"
```
Options : `--start 61.5` (forcer le refrain, en secondes), `--stage-bars "2,4,4,4,4"` (mesures par étape),
`--full-bars 8` (mesures de FULL SONG), `--max-duration 75`, `--stems dossier/` (tes propres pistes `vocals`/`other`/`bass`/`drums`),
`--min-duration 62`, `--accent "#9FB4FF"`, `--cta "..."`, `--cta2 "..."`.

Mode Échappée : `--mode escape --rings 500 --pre-bars 16` (et `--drop-bars 16` pour fixer la longueur du refrain).

## Droits d'auteur

Si tu utilises un son du commerce, TikTok peut couper le son de la vidéo ou la bloquer. Le plus sûr :
un son de la bibliothèque TikTok, de la musique libre de droits, ou tes propres productions.

La police Inter (dossier `fonts/`) est sous licence SIL Open Font License 1.1 (voir `fonts/OFL.txt`).
