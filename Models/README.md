## Underwater_semantic_segmentation for SONAR data

### Test various architectures on the underwater sonar dataset and compare the results
    Benchmarking various semantic segmentation on the dataset. Various architectures (listed below) are used with various encoders (ResNet18, ResNet34, ResNet50, VGG16 etc.)



* Architectues Used 

    >    [UNet](https://arxiv.org/pdf/1505.04597.pdf)

    >    [LinkNet](https://arxiv.org/abs/1707.03718)

    >    [PSPNet](https://arxiv.org/abs/1612.01105)

    >    [DeepLab](https://arxiv.org/pdf/1606.00915.pdf) 

* Library used 
    >[https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)

* Encoders used
    *   [ResNet](https://arxiv.org/abs/1512.03385)
        >       ResNet18
        >       ResNet34
        >       ResNet50
        >       ResNet101
        >       ResNet152

    *   [Vgg](https://arxiv.org/abs/1409.1556)
        >       Vgg16
        >       Vgg19

    *   [InceptionNet](https://arxiv.org/pdf/1409.4842.pdf)
        >       InceptionNetV3
        >       InceptionResNetV2

    *   [EfficientNet](https://arxiv.org/abs/1905.11946)
        >       b0
        >       b1
        >       b2

* Dataset
    * [Water Tank Marine Debris with ARIS Explorer 3000](https://github.com/mvaldenegro/marine-debris-fls-datasets/releases/tag/watertank-v1.0)

Sample image with labelling is as shown -: ![](img.png)


* Classes
    >        0:  Background         
    >        1:  Bottle         
    >        2:  Can            
    >        3:  Chain          
    >        4:  Drink-carton   
    >        5:  Hook           
    >        6:  Propeller      
    >        7:  Shampoo-bottle 
    >        8:  Standing-bottle
    >        9:  Tire           
    >        10: Valve         
    >        11: Wall   

