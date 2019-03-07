model = pretrainedmodels.__dict__['densenet121'](num_classes=1000, pretrained='imagenet') 
dim_feats = model.last_linear.in_features 
model.last_linear = nn.Sequential(nn.Dropout(args.dropout_p), nn.Linear(dim_feats, args.nb_classes))
model.train().cuda()  