# *LyricCovers*: A comprehensive large-scale dataset of cover songs with lyrics

This repository contains the code for the Paper "*LyricCovers*: A comprehensive large-scale dataset of cover songs with lyrics".

## Abstract

This research offers a detailed examination of a novel dataset that collates original musical compositions alongside their derivative cover versions. Unique in its inclusion of both audio files as well as annotated lyrical content, the dataset enlists more than 70,000 tracks, encompassing more than 18,000 cover song groupings. It stands as the most diverse compendium of cover songs currently available for study. The characteristics of the \textit{LyricsCovers} dataset are thoroughly analyzed through its metadata, and empirical evaluations in the subsequent experimental lyrics analysis section suggest that lyrical analysis is a fundamental component in identifying and studying cover songs.



## Setup

Install dependencies with [poetry](https://python-poetry.org/), the recommended version is `1.7.1`:

```bash
poetry install
```

Create a `.env` file with the following content:

```bash
HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
PROJECT_ID=YOUR_GOOGLE_CLOUD_PROJECT_ID
TOKENIZERS_PARALLELISM=false
```

## Usage

Everything is in organized in ZenML Pipelines and meant to be run in the Google Cloud. To run the pipelines, you need to set up a [ZenML server](https://docs.zenml.io/deploying-zenml/zenml-self-hosted) and at least a [Google Cloud Function](https://cloud.google.com/functions/docs) with [Pup/Sub](https://cloud.google.com/pubsub?hl=de) and a [Google Cloud Storage](https://cloud.google.com/storage/docs) to automatically download the audio files.

1) Create a Google Cloud Pub/Sub topic called `youtube` for the Cloud Run subscription.
2) Create a second Google Cloud Pub/Sub topic called `download-video-response` for the Cloud Run response.
3) Create a Google Cloud Function with the code of `cloud_functions`, add a Pub/Sub trigger for the `youtube` topic. You need to set the Project ID and the Bucket Name in the `main.py` file.
4) Create a Google Cloud Bucket where the audio files will be stored
6) Setup your ZenML server
7) (Optional) Create a [ZenML Google Cloud Stack](https://docs.zenml.io/v/0.56.2/user-guide/cloud-guide/gcp-guide)
8) Create a docker base image for the source separation with the `dockerfile` in the `docker` folder. If you run the pipeline in the Google Cloud you need to push the image to the Google Cloud Container Registry.
9) Use the `python run.py` script to run the pipelines, you can use the `--help` flag to see the available options

## Interative plot

Open the [Interactive plot](assets/genres.html)


### Current dataset

```
<class 'pandas.core.frame.DataFrame'>
Index: 77751 entries, 69 to 3936833
Data columns (total 32 columns):
 #   Column                              Non-Null Count  Dtype 
---  ------                              --------------  ----- 
 0   id                                  77751 non-null  int64 
 1   url                                 77751 non-null  object
 2   title                               77751 non-null  object
 3   artist                              77751 non-null  object
 4   artist_id                           77751 non-null  int64 
 5   language                            77507 non-null  object
 6   lyrics                              77751 non-null  object
 7   lyrics_state                        77751 non-null  object
 8   youtube_url                         77751 non-null  object
 9   youtube_type                        77751 non-null  object
 10  spotify_url                         5378 non-null   object
 11  spotify_type                        5378 non-null   object
 12  soundcloud_url                      5596 non-null   object
 13  soundcloud_type                     5596 non-null   object
 14  original_id                         77751 non-null  int64 
 15  is_cover                            77751 non-null  bool  
 16  vevo_url                            3 non-null      object
 17  vevo_type                           3 non-null      object
 18  soundfile_available                 77751 non-null  bool  
 19  tags                                77750 non-null  object
 20  youtube_download_status             70527 non-null  object
 21  youtube_download_gs_path            69875 non-null  object
 22  source_separation_status_demucs     69875 non-null  object
 23  vocals_demucs                       69875 non-null  object
 24  drums_demucs                        69875 non-null  object
 25  bass_demucs                         69875 non-null  object
 26  other_demucs                        69875 non-null  object
 27  transcription_status_demucs_w_tiny  69875 non-null  object
 28  transcription_demucs_w_tiny         69875 non-null  object
 29  source_separation_status_spleeter   69875 non-null  object
 30  vocals_spleeter                     69875 non-null  object
 31  accompaniment_spleeter              69875 non-null  object
dtypes: bool(2), int64(3), object(27)
memory usage: 18.5+ MB
```