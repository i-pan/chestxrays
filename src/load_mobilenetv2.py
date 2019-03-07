import os 
WDIR = os.path.dirname(os.path.abspath(__file__))
import sys ; sys.path.insert(0, WDIR)
from mobilenetv2 import MobileNetV2 
model = MobileNetV2(num_class=1000) 
model.load_state_dict(torch.load('mobilenetv2.pth'))
model.classifier = nn.Sequential(nn.Dropout(args.dropout_p), nn.Linear(1280, args.nb_classes))
model.train().cuda()