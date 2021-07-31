#!/usr/bin/env python
# coding: utf-8

from ae_feature_extractor import autoencoder


ae = autoencoder ()
ae.train_model_c ()
ae.extract_features()

