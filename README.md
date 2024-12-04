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
Index: 78862 entries, 69 to 3936785
Data columns (total 30 columns):
 #   Column                             Non-Null Count  Dtype         
---  ------                             --------------  -----         
 0   id                                 78862 non-null  int64         
 1   url                                78862 non-null  object        
 2   title                              78862 non-null  object        
 3   artist                             78862 non-null  object        
 4   artist_id                          78862 non-null  int64         
 5   language                           78003 non-null  object        
 6   lyrics                             78829 non-null  object        
 7   lyrics_state                       78862 non-null  object        
 8   youtube_url                        78862 non-null  object        
 9   youtube_type                       78862 non-null  object        
 10  spotify_url                        7723 non-null   object        
 11  spotify_type                       7723 non-null   object        
 12  soundcloud_url                     7258 non-null   object        
 13  soundcloud_type                    7258 non-null   object        
 14  original_id                        78862 non-null  int64         
 15  is_cover                           78862 non-null  bool          
 16  vevo_url                           9 non-null      object        
 17  vevo_type                          9 non-null      object        
 18  youtube_download_status            78862 non-null  object        
 19  youtube_download_gs_path           78862 non-null  object        
 20  source_separation_status_htdemucs  78862 non-null  object        
 21  vocals_htdemucs                    78830 non-null  object        
 22  drums_htdemucs                     0 non-null      object        
 23  bass_htdemucs                      0 non-null      object        
 24  other_htdemucs                     0 non-null      object        
 25  tags                               78861 non-null  object        
 26  release_date                       59457 non-null  datetime64[ns]
 27  release_year                       59457 non-null  datetime64[ns]
 28  transcription_status               17001 non-null  object        
 29  transcription                      17001 non-null  object        
dtypes: bool(1), datetime64[ns](2), int64(3), object(24)
memory usage: 18.1+ MB
```
