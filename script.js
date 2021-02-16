/*!
 * script.js v1.0
 *
 * Copyright © 2021 yuichiro hiramoto All Rights Reserved.
 * Copyright © 2021-2021 yuichiro hiramoto All Rights Reserved.
 */

class InstanceNormalization extends tf.layers.Layer {
    static className = 'InstanceNormalization';
    constructor(config) {
        super(config);
        this.trainable;
        this.axis=config.axis;
        this.epsilon=config.epsilon;
        this.center=config.center;
        this.scale=config.scale;
        this.beta_initializer=config.beta_initializer;//'zeros';
        this.gamma_initializer=config.gamma_initializer;//'ones';
        this.beta_regularizer=config.beta_regularizer;
        this.gamma_regularizer=config.gamma_regularizer;
        this.beta_constraint=config.beta_constraint;
        this.gamma_constraint=config.gamma_constraint;
    }

    build(inputShape) {
        /*const ndim = inputShape.length;
        console.log("Build:inputShape/ndim:")
        console.log(inputShape)
        console.log(ndim)*/

        var shape;
        if (this.axis==null){
            shape = [1]
        }else{
            shape=tf.tensor([inputShape[this.axis]])
        }
        /*console.log("SHAPE:")
        console.log(shape)*/
        
        if (this.scale){
            this.gamma = this.addWeight('gamma',shape,
                                        'float32',
                                        tf.initializers.ones(),
                                        this.gamma_regularizer,
                                        this.trainable,
                                        this.gamma_constraint)
        }else{
            this.gamma=null;
        }

        if (this.center){
            this.beta = this.addWeight('beta',shape,
                                        'float32',
                                        tf.initializers.zeros(),
                                        this.beta_regularizer,
                                        this.trainable,
                                        this.beta_constraint)
        }else{
            this.beta = null;
        }
    }

    call(input) {
        return tf.tidy(() => {
            const input_shape = input[0].shape;
            const reduction_axes = input_shape.length-1;

            var mean = tf.mean(input[0], reduction_axes,true)
            var stddev = tf.moments(input[0], reduction_axes,true).variance.sqrt().add(this.epsilon)
            var normed = ((input[0]).sub(mean)).div(stddev);
            /*console.log("mean,stddev,normed->")
            console.log(mean)
            console.log(mean.shape)
            console.log(stddev)
            console.log(stddev.shape)
            console.log(normed)
            console.log(normed.shape)*/

            var broadcast_shape = [];//[1] * (input_shape.length)
            for(var i=0;i<input_shape.length;i++){
                broadcast_shape.push(1);
            }

            if (this.axis != null){
                broadcast_shape[this.axis] = input_shape[this.axis]
            }
            /*console.log(broadcast_shape)                      
            console.log(broadcast_shape.shape)
            console.log(this.gamma.read())
            console.log(this.gamma.read().shape)*/
            var broadcast_gamma;
            var broadcast_beta;
            if (this.scale){
                broadcast_gamma = tf.reshape(this.gamma.read(), broadcast_shape)
                normed = normed.mul(broadcast_gamma)
            }

            if (this.center){
                broadcast_beta = tf.reshape(this.beta.read(), broadcast_shape)
                normed = normed.add(broadcast_beta)
            }
            return normed
        });
    }

    getConfig() {
        const config = super.getConfig();
        return config;
    }

    /**
     * The static className getter is required by the 
     * registration step (see below).
     */
    static get className() {
        return 'InstanceNormalization';
    }
}
/**
 * Regsiter the custom layer, so TensorFlow.js knows what class constructor
 * to call when deserializing an saved instance of the custom layer.
 */
tf.serialization.registerClass(InstanceNormalization);

//参考：https://tech-it.r-net.info/program/javascript/265/
//
// 画像の取得
var img = document.getElementById('original');
var imgReader;
var progress_bar = document.getElementById('progress_bar');

var canvas = document.createElement('canvas');

const fileup = (e) => {
    console.log(e)
    //var img = document.getElementById('original');
    const reader = new FileReader();
    imgReader = new Image();
    reader.onloadend = () => {
        imgReader.onload = () => {
            const imgType = imgReader.src.substring(5, imgReader.src.indexOf(';'));
            //const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = imgReader.width;
            canvas.height = imgReader.height;
            ctx.drawImage(imgReader,0,0,imgReader.width,imgReader.height);
            console.log(imgReader.width)
            console.log(imgReader.height)
            img.src = canvas.toDataURL(canvas);
        }
        imgReader.src = reader.result;
        progress_bar.setAttribute("style", "width:10%");
        console.log("10%");
    }
    reader.readAsDataURL(e.files[0]);
    app()
}

const progress = async () =>{
    console.log("30%");
    progress_bar.setAttribute("style", "width:30%");
}

const app = async () => {    
    document.getElementById('isConvert').innerText='変換中です。';
    
    // モデルの読み込み
    await faceapi.nets.tinyFaceDetector.load("models/");
    progress();
    

    // 顔検出の実行
    //const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
    const displaySize = { width: img.width, height: img.height }
    // resize the overlay canvas to the input dimensions
    const canvas = document.createElement('canvas')
    faceapi.matchDimensions(canvas, displaySize)

    const startTime = Date.now(); // 開始時間
    
    /* Display detected face bounding boxes */
    //const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
    const detections = await faceapi.detectSingleFace(imgReader, new faceapi.TinyFaceDetectorOptions())
    const endTime = Date.now(); // 終了時間
    console.log(endTime - startTime); // 何ミリ秒かかったかを表示する
    progress_bar.setAttribute("style", "width:50%");
    console.log("50%");
    //faceapi.dispose();
    //detections.dispose();

    var h = parseInt(detections._box._height)
    var w = parseInt(detections._box._width)
    var x = parseInt(detections._box._x)
    var y = parseInt(detections._box._y)

    // draw detections into the canvas
    //faceapi.draw.drawDetections(canvas, resizedDetections)
    
    //get tensor from canvas
    var tensor = GetTensorFromCanvas(x,y,w,h)

    document.getElementById('isConvert').innerText='変換中です。もうすぐ完成！';
    document.getElementById('isConvert').innerText='60%';
    
    progress_bar.setAttribute("style", "width:60%");
    console.log("60%");
    //Load Model
    model = await tf.loadLayersModel('tfmodel/model.json')
    document.getElementById('isConvert').innerText='70%';
    progress_bar.setAttribute("style", "width:70%");
    console.log("70%");
    //Inference
    
    console.log(tf.memory())
    console.log("InferenceStart")
    var prediction = await model.predict(tensor)
    document.getElementById('isConvert').innerText='90%';
    tensor.dispose();
    model.dispose();
    progress_bar.setAttribute("style", "width:90%");
    console.log("90%");

    /*console.log("Prediction,PredictionShape,Unstack(0)[0]=>")
    console.log(prediction)
    console.log(prediction.shape)
    console.log(prediction.unstack(0)[0])*/

    //prediction=prediction.unstack(0)[0]
    let offset_mul = tf.scalar(127.5);
    let offset_add = tf.scalar(1);
    console.log("A");
    prediction=prediction.unstack(0)[0].add(offset_add).mul(offset_mul);
    console.log("B");
    var image = prediction.clipByValue(0,255).toInt();
    prediction.dispose();
    console.log("C");
    

    //var temp = document.createElement("canvas");
    console.log("D");
    console.log(tf.memory())
    tf.tidy(()=>{
        await tf.browser.toPixels(image.resizeNearestNeighbor([128,128]), canvas);
    })
    
    image.dispose();
    console.log(tf.memory())
    console.log("E");
    console.log(tf.memory())
    var png = canvas.toDataURL();
    console.log("F");
    document.getElementById('translated').src=png;
    console.log("G");
    document.getElementById('isConvert').innerText='完成しました！';
    progress_bar.setAttribute("style", "width:100%");

    console.log("Prediction ended.")
}

function GetTensorFromCanvas(x,y,w,h) {
    //描画コンテキストの取得
    //var canvas = document.getElementById('sample');
    canvas.width=w;
    canvas.height=h;
    if (canvas.getContext) {
        var context = canvas.getContext('2d');
        //元イメージの座標(x, y)から幅w高さhの範囲を使用して、
        //座標(0, 0)の位置に、サイズw×hでイメージを表示
        context.drawImage(imgReader, x, y, w, h , 0, 0, w, h);
    }
    var faceImg=document.getElementById("face");
    faceImg.src = canvas.toDataURL(canvas);
    tensor_image=preprocessImage(canvas);
    return tensor_image;
}

function preprocessImage(image){
    //オフセット等を学習時と一致させる。
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([128,128]).toFloat();
    let offset_div = tf.scalar(127.5);
    let offset_sub = tf.scalar(1);
    return tensor.div(offset_div).sub(offset_sub).expandDims();
}