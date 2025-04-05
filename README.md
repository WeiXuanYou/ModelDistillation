# Introduction
Using model distillation, the RegNet model was successfully compressed from 83M parameters down to approximately 1M (achieving an 83x compression ratio). To maintain model performance, Contrastive Representation Distillation (CRD) was employed for optimization, supplemented by K-Fold cross-validation and tuning of distillation temperature/alpha parameters, which particularly enhanced performance in limited-data scenarios. Ultimately, the student model's accuracy experienced only a slight decrease (training: 0.67 -> 0.65; validation: 0.71 -> 0.67), while significantly reducing hardware resource dependency and accelerating model inference speed.
- Using model distillation, the RegNet model was successfully compressed from 83M parameters down to approximately 1M (achieving an 83x compression ratio).
