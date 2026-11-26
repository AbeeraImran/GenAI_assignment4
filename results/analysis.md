# Analysis

## E.1 MAE: why mask 75%?

Neighbouring pixels in a satellite tile are very similar, so a hidden patch can be guessed by copying its neighbours. With only 15% masked, the model would learn almost nothing about the scene. BERT can mask 15% of a sentence because every word carries a lot of meaning and is hard to guess. With 75% masked, only 16 of 64 patches are visible, so the model has to learn what fields, roads, water and buildings look like and where they usually are. In `mae_recon.png` the reconstructions are blurry and blocky but keep the overall colours and layout. For example, the pale river band in the last tile comes back as a pale diagonal band. Fine details are lost, which is expected from a pixel MSE loss.

## E.2 MAE: which classes are confused?

The most confused pairs (true, predicted) were AnnualCrop to PermanentCrop (170), SeaLake to Pasture (150), River to Highway (128), PermanentCrop to HerbaceousVegetation (124) and Residential to HerbaceousVegetation (115). The crop and vegetation classes are all green or brown field textures seen from above, so they look alike at 64 by 64 pixels. River and Highway are both thin, winding strips across a plain background. Residential tiles often contain a lot of green, which pulls them towards vegetation. SeaLake to Pasture is the least obvious one. Both can be smooth areas of one colour, and the frozen feature is an average over all patches, so texture detail is washed out. I did not investigate this further. The overall accuracy is only 54%, and the classifier loss was still falling at epoch 200 (1.13), so part of the confusion is underfitting from using 160 labelled images and the prescribed 200 epochs.

## E.3 VAE: reconstruction and KL

The reconstruction term fell from about 204 to 105 nats on the training set, while the KL term rose from about 15.6 to 25.4 and then levelled off. The model first learned to reconstruct, which needs the latent code to carry more information, and that costs KL. The test curves follow the training curves closely, so there is no sign of overfitting. The final test negative ELBO is 131.03 (105.54 reconstruction + 25.49 KL). If the KL term were removed, the model would become a plain autoencoder. The encoder would be free to place codes anywhere and make sigma very small, and nothing would keep the latent space close to N(0, I). Samples drawn from N(0, I) would then land in regions the decoder never saw, so I expect mostly meaningless or unrecognisable images.

## E.4 VAE versus diffusion

The diffusion samples are sharper than the VAE samples. The VAE decoder is trained with a pixel-wise Bernoulli likelihood, and when it is unsure about a stroke it outputs a grey average, which looks blurry. The 16-dimensional latent bottleneck also removes detail. The diffusion model is trained to predict the noise at many noise levels, and sampling takes 1,000 small denoising steps, so each step only has to make a small correction and details can be refined step by step. The cost is that sampling takes 1,000 network passes, while the VAE needs one. I did not use numbers from the lecture, so this explanation should be read together with the lecture's comparison of the two objectives.

## E.5 Diffusion: which characters are worst?

From the 40 by 8 grid, the rows that look least consistent are gaaf, hamza, rhraa and zwaad, and one qaaf sample is an almost blank grey square, which is a failed sample. Many Urdu letters share the same base shape and differ only by dots, for example ayn and ghain, jeem, cheey, haa and khaa, zaal and daal, and swaad and zwaad. At 28 by 28 pixels these dots are only one or two pixels, so the model often draws the right shape with the wrong dots, and the result looks like a neighbouring character. This also explains the low blind-test score (1 of 40): a reader cannot tell dot-only differences apart in tiny samples, and the person labelling was not certain about the real characters either. I did not run the same test on real images, so I cannot say how much of the low score comes from the model and how much from the difficulty of the task.

## E.6 RAG: retrieval failures

98 of 707 test questions (13.9%) had no relevant abstract in the top 10. Three examples:

- "Is capmatinib effective for glioblastoma?" The relevant abstract starts with a conflict-of-interest statement, so its text has little to do with the question. The retriever returned a glioma chemotherapy abstract.
- "What is F105-P?" The relevant abstract defines a protamine-antibody fusion protein. The retriever returned an abstract about FOP (Fibrodysplasia Ossificans Progressiva). A rare code name is split into small word pieces with little meaning for the embedding model, and dense retrieval has no exact-match step, so I think it matched a similar-looking acronym.
- "Which main viral protein is targeted by the drug remdesivir?" The relevant abstract is a clinical analysis of Ebola trials, and it does not talk about a target protein. The retriever returned an abstract about liver transporters. The question asks about a mechanism, and the relevant abstract is about a trial.

The top-5 chunks always come from the top-ranked abstracts, so when retrieval failed none of the five passages came from a relevant abstract. Retrieval failed for 14 of the first 100 questions, and the model said it did not know for 13 of the first 100. For the capmatinib question the model still gave a confident answer ("Yes, capmatinib is effective for glioblastoma. [3]"). Since no passage came from a relevant abstract, the answer is not backed by the evidence BioASQ marks as relevant (passage [3] is shown in `samples.md`, question 1). In my own reading of 20 answers, 15 were supported and 15 were correct. The CCR4 question was neither: the model answered "CCR4 antagonists", the passages never name a CCR4 drug, and the reference answer is mogamulizumab.

## E.7 Why is Recall@k lower than Hit@k?

Hit@k only asks whether at least one relevant abstract is in the top k. Recall@k asks what fraction of the relevant abstracts was found, out of min(k, |R|). Questions have 8.66 relevant abstracts on average, so finding one of them is much easier than finding many. Hit@k is always at least as large as Recall@k, and they are equal at k = 1, as in my table (0.622 for both). My Recall@k curve first drops from 0.622 at k = 1 to about 0.586 at k = 5 and then rises to about 0.638 at k = 20. The denominator min(k, |R|) grows with k at first, faster than the number of relevant abstracts found, and later the retriever finds more of them than the denominator adds.
