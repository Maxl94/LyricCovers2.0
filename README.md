# LyricCovers 2.0: An enhanced dataset for cover song analysis

This repository contains the code for the Paper "*LyricCovers*: A comprehensive large-scale dataset of cover songs with lyrics".

## Abstract

This research offers a detailed examination of a novel dataset that collates original musical compositions alongside their derivative cover versions. Unique in its inclusion of both links to YouTube as well as and lyrical content, the dataset enlists more than 78,000 tracks, encompassing more than 24,000 cover song groupings. It stands as the most diverse compendium of cover songs currently available for study. The characteristics of the LyricCovers dataset are thoroughly analyzed through its metadata, and empirical evaluations in the subsequent experimental lyrics analysis section suggest that lyrical analysis is a fundamental component in the identification and study of cover songs. This work presents a baseline approach to cover song detection, with an emphasis on lyrical content processing. It describes the extraction of lyrics from the audio files and the application of the Jina Embeddings 2 Model, fine-tuned with a hard triplet-loss objective, which successfully exploits lyric similarity to accurately identify cover songs.



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

The dataset are part of this repository and can be found in the `data` folder. 

The main dataset is a pandas DataFrame with the following columns:


```
<class 'pandas.core.frame.DataFrame'>
Index: 78862 entries, 69 to 3936785
Data columns (total 28 columns):
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
 19  tags                               78861 non-null  object        
 20  release_date                       59457 non-null  datetime64[ns]
 21  release_year                       59457 non-null  datetime64[ns]
dtypes: bool(1), datetime64[ns](2), int64(3), object(22)
memory usage: 16.9+ MB
```

The train, validation and test datasets are stored in the `data` folder as well.

## Usage & Evaluation Guidelines

### Dataset Usage
The dataset is split into training, validation, and test sets, provided in the `data` folder. We recommend using the training and validation sets for model development and hyperparameter tuning. There are no strict constraints on how to utilize these portions of the dataset - researchers and developers are free to experiment with different approaches based on their specific requirements.

### Evaluation Protocol
For evaluating cover song detection systems, we strongly recommend following a practical evaluation protocol:

1. Split the test dataset into two components:
   - A database containing all original songs
   - A query set containing the cover versions

2. Implementation flow:
   - Calculate cross-similarity between each query (cover) and all songs in the database
   - Rank the database songs based on their similarity scores
   - Evaluate the system using standard retrieval metrics (e.g., Mean Reciprocal Rank, Mean Average Precision)

This evaluation approach mirrors real-world scenarios where systems need to identify whether a given song is a cover version by comparing it against a database of original recordings. It provides a more practical and meaningful assessment of a model's performance in cover song detection tasks.

We reco

## Citation

```bibtex
@article{balluff2022lyriccovers,
    title={LyricCovers 2.0: An Enhanced Dataset for Cover Song Analysis},
    author={Balluff, Maximilian and Auch, Maximilian and Mandl, Peter and Wolff, Christian},
    journal={IADIS International Journal on WWW/Internet},
    volume={22},
    number={2},
    pages={75--92},
    year={2022},
    issn={1645-7641}
}
@inproceedings{Balluff2024LyricCovers,
    title={LyricCovers: A comprehensive large-scale dataset of cover songs with lyrics},
    author={Maximilian Balluff, Peter Mandl, Andreas Wolff},
    booktitle = {Proceedings of the International Conferences on Applied Computing \& WWW/Internet},
    year={2024},
    editor = {Miranda, Paula and Isaías, Pedro},
    month = {October}
}
```
