def build_multires_model():
    # 모델의 input shape을 결정합니다.
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    
    # backbone 모델로부터 feature map을 가져옵니다.
    features = backbone_features(inputs)
    
    # 사용할 feature map을 여러개 고르고, 각 feature map을 같은 크기로 upsampling 해줍니다.
    upsample1 = tf.keras.layers.UpSampling2D(
        size=(4, 4), interpolation='bilinear')
    
    x1 = upsample1(features[4])
    
    upsample2 = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation='bilinear')
    
    x2 = upsample2(features[3])
    x3 = features[2]
    
    # 같은 크기로 만들어진 feature map을 합쳐줍니다.
    concat = tf.keras.layers.Concatenate(axis=-1)
    x = concat([x1, x2, x3])
    
    # 해당 feature map을 기반으로 segmentation prediction mask를 만들어주는 CNN을 만들어줍니다.
    convnet = tf.keras.Sequential(layers=[
        tf.keras.layers.Conv2D(16, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(OUTPUT_CHANNELS, 3, padding='same'),
    ])
    x = convnet(x)
    
    # 출력값을 bilinear interpolation을 이용해서 128x128 크기로 만들어줍니다.
    upsample = tf.keras.layers.UpSampling2D(
        size=(8, 8), interpolation='bilinear')
    
    x = upsample(x)
    
    # x를 최종 출력값으로 가지는 모델을 만들어줍니다.
    return tf.keras.Model(inputs=inputs, outputs=x)
