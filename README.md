# Language-Adapters

This repo accompanies our team's research on data-efficient deep learning approaches to translating low-resource languages — in particular, the experimentation related to language adapters.

The folder _adapter-transformers_ is an extension of the Adapter-hub library for use in our research into the use of monolingual adapters for multilingual machine translation. This module extends support of the adapter-hub library to include the M2M-100 model from Meta, and also facilitates the use of monolingual adapters (adapters only in the encoder or only in the decoder block), and to allow multiple adapters to be active at the same time (the _Pair()_ composition class). For more information about monolingual adapters, refer to the paper ["Monolingual Adapters for Zero-Shot Neural Machine Translation" (Philip et al. 2020)](https://aclanthology.org/2020.emnlp-main.361/).

The folder _Data_ stores our data for this project. The train and dev files consist of a mix of **MAFAND-MT** and **WebCrawl African** data for English, Yoruba, Hausa, Swahili, and Igbo. Because we are researching techniques for low resource language machine translation, we randomly selected only a subset of the available data for Swahili in order to have comparable dataset sizes between each of the EN-X language directions. The test set files consist of **Flores 200** data for each language direction.

The _m2m_100_training_lang_adapters.ipynb_ notebook uses this _adapter-transformers_ module to add EN-HA language adapters (ie. an English encoder adapter and a Hausa decoder adapter) to an M2M-100 model, and train and evaluate those adapters.

Finally, the _final_poster_ reports our team's completed research on the use of language adapters and reinforcement learning for data- and parameter-efficient machine translation.
