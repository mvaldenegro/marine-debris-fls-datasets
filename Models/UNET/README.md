## UNET architecture with multiple encoders

### Training parameters
TRAINING_IMGS   = 1000
VALIDATION_IMGS = 251
TESTING_IMGS    = 617

EPOCHS     = 30
BATCH_SIZE = 16

ENCODERS -> ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, VGG16, VGG19, InceptionV2, InceptionResNetV3, EfficientNetb0, EfficientNetb1