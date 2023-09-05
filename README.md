# Solar_France - Solar Panel Detection Using Remote Sensing Imagery with Transfer Learning

Welcome to the repository dedicated to my master's thesis, "Exploring the Effectiveness of Transfer Learning through Data Combination and Augmentation in Solar Panel Detection Using Remote Sensing Imagery".

 !(https://github.com/VenturaHaze/Solar_France/blob/b10afc533674a4496456138f3c04eb34fa8ca861/UNet_pretrained100_Epoch10_pred1.png)


## Introduction

This research unravels the intricacies and efficiencies of diverse training strategies in the field of remote sensing, with a prime focus on Transfer Learning (TL). In an era where the sustainable energy sector is surging ahead, accurate and efficient detection of solar installations becomes pivotal. Harnessing aerial imagery to detect these installations introduces a plethora of challenges, predominantly stemming from the variability in image data across geographies.
Purpose

The primary purpose of this repository is to offer a comprehensive look at the methodologies, data sources, and results from the research. By diving deep into techniques like data amalgamation from multiple global regions, the Sequential Training (ST) paradigm, and a unique "solar snippet" method for data augmentation, this research showcases a blend of innovation and rigorous empirical study.

## Context: The Data Landscape

Central to this research is the DeepStat WP5 Solar Panel Dataset by CBS, a rich collection of aerial photographs from the Limburg region in the Netherlands. This dataset, combined with others from varied geographies, forms the backbone of the analytical efforts.

Annotations play a pivotal role, particularly when the goal is semantic segmentation. The dataset from CBS was meticulously augmented with annotations crafted by a team of dedicated volunteers. This herculean annotation effort added another dimension to the dataset, laying the foundation for more nuanced research on solar panel detection.

Data processing and transformation, both pre and post, form a crucial segment of the workflow. It begins with careful data retrieval, ensuring a constant ground sampling distance, and proceeds to augment the dataset with varied transformations. These transformations, from random crops to color adjustments, aim to strengthen the model's robustness across varied data conditions.

## Key Findings

Data Diversity: Harnessing geographically diverse datasets significantly bolsters model generalization.

Sequential Training (ST): The ST paradigm emerges as an impactful methodology for solar panel segmentation.

Data Augmentation with "Solar Snippet": The introduction of this technique... (Continue elaborating.)

## Acknowledgements

A journey of this magnitude owes much to countless contributors. Immense gratitude to my advisors, collaborators, and the ever-giving open-source community. 
