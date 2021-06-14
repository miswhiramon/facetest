/*!
 * script.js v1.0
 *
 * Copyright © 2021 Nd All Rights Reserved.
 * Copyright © 2021-2021 Nd All Rights Reserved.
 */

class InstanceNormalization extends tf.layers.Layer {
    //static className='InstanceNormalization';
    constructor(config) {
        super(config);
        this.className='InstanceNormalization';
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
        var shape;
        if (this.axis==null){
            shape = [1]
        }else{
            shape=tf.tensor([inputShape[this.axis]])
        }
        
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
            var mean = tf.mean(input[0], [1,2],true);
            var stddev = tf.moments(input[0], [1,2],true).variance.sqrt().add(this.epsilon)
            var normed = ((input[0]).sub(mean)).div(stddev);


            var broadcast_shape = []; //[1] * (input_shape.length)
            for(var i=0;i<input_shape.length;i++){
                broadcast_shape.push(1);
            }

            if (this.axis != null){
                broadcast_shape[this.axis] = input_shape[this.axis]
            }

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


//tensorflow jsの実行環境の判定
var ut = navigator.userAgent;
console.log(ut);
if(ut.indexOf('iPhone') > 0 || ut.indexOf('iPod') > 0 || ut.indexOf('iPad') > 0 ){
    console.log('iPhone iPod iPad');
    tf.setBackend('cpu');
    console.log(tf.getBackend());
}else{
    console.log('Windows Mac Android');
    tf.setBackend('webgl');
    console.log(tf.getBackend());
}


// 画像の取得
var img = document.getElementById('original');
var imgReader;
var progress_bar= document.getElementById('progress_bar'); 

//ボタンの取得
var download_button=document.getElementById('download');

var canvas = document.createElement('canvas');

//ギャラリーアクセス
const fileup = (e) => {
    progress_bar= document.getElementById('progress_bar');
    progress_bar.setAttribute("style", "width:0%");
    img = document.getElementById('original');
    canvas = document.createElement('canvas');
    const reader = new FileReader();
    imgReader = new Image();
    reader.onloadend = () => {
        imgReader.onload = () => {
            const ctx = canvas.getContext('2d');
            canvas.width = imgReader.width;
            canvas.height = imgReader.height;
            ctx.drawImage(imgReader,0,0,imgReader.width,imgReader.height);
            console.log(imgReader.width)
            console.log(imgReader.height)
            img.src = canvas.toDataURL(canvas);
        }
        imgReader.src = reader.result;
        console.log(imgReader);
        progress_bar.setAttribute("style", "width:10%");
        progress_bar.innerText="10%";
        console.log("10%");
        app()
    }
    reader.readAsDataURL(e.files[0]);
    
}

//カメラアクセス
//Camera
function startCamera(){
    const div1 = document.getElementById("camera");
    const camera_button = document.getElementById('camera_button');
    // 要素の追加
    if (!div1.hasChildNodes()){
        const p1 = document.createElement("video");
        p1.setAttribute("id","myVideo");
        p1.setAttribute("autoplay","1");

        var constraints = { audio: false, video: { facingMode: "user" } };

        navigator.mediaDevices.getUserMedia( constraints )
        .then(
        function( stream ) {
          div1.appendChild(p1);
          var video = document.querySelector( 'video' );
          video.srcObject = stream;
          video.onloadedmetadata = function( e ) {
          video.play();

          const button_field=document.getElementById('camera_button');
          button_field.style.width=video.clientWidth+"px";
          console.log(button_field);
          };

          if (!camera_button.hasChildNodes()){
            camera_button.innerHTML="<div class='photo-button' onclick='stopCamera()'><div class='circle'></div><div class='ring'></div></div>"
          }
        })
    }
}

function stopCamera(){
    //進捗ゲージ初期化
    progress_bar= document.getElementById('progress_bar');
    progress_bar.setAttribute("style", "width:0%");
    //要素取得
    const div1 = document.getElementById("camera");
    const camera_button = document.getElementById('camera_button');
    img = document.getElementById('original');
    //imgReader = new Image();
    if (div1.hasChildNodes()){
        //get video Element
        const video=document.getElementById('myVideo');
        //create temp canvas
        const temp_canvas=document.createElement("canvas");
        const ctx=temp_canvas.getContext("2d");
        //canvas resize
        temp_canvas.width=video.videoWidth;
        temp_canvas.height=video.videoHeight;
        //draw videoImage to Canvas
        ctx.drawImage(video,0,0,video.videoWidth,video.videoHeight);
        //img draw
        img.src=temp_canvas.toDataURL();
        imgReader=img;
        console.log(imgReader);

        //カメラの終了
        const tracks = document.getElementById('myVideo').srcObject.getTracks();        
        tracks.forEach(track => {
            track.stop();//カメラの停止
        });
        document.getElementById('myVideo').srcObject = null;
        div1.removeChild(div1.firstChild);
        app()
    }
    if (camera_button.hasChildNodes()){
        camera_button.removeChild(camera_button.firstChild);
    }
}

//顔検出・変換処理
const app = async () => {    
    document.getElementById('isConvert').innerText='顔部分検出中です。';
    
    // モデルの読み込み
    await faceapi.nets.tinyFaceDetector.load("models/");
    progress_bar.setAttribute("style", "width:30%");
    progress_bar.innerText="30%";
    

    // 顔検出の実行
    //const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
    //const displaySize = { width: img.width, height: img.height }
    // resize the overlay canvas to the input dimensions
    //const canvas = document.createElement('canvas')
    //faceapi.matchDimensions(canvas, displaySize)

    var startTime = Date.now(); // 開始時間
    
    /* Display detected face bounding boxes */
    //const detections = await faceapi.detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
    const detections = await faceapi.detectSingleFace(imgReader, new faceapi.TinyFaceDetectorOptions())
    var endTime = Date.now(); // 終了時間
    console.log("FaceDetectTime:"+(endTime - startTime)); // 何ミリ秒かかったかを表示する
    if(detections==null){
        document.getElementById('isConvert').innerText='顔が検出できませんでした。';
        return
    }
    progress_bar.setAttribute("style", "width:40%");
    progress_bar.innerText="40%";

    var h = parseInt(detections._box._height)
    var w = parseInt(detections._box._width)
    var x = parseInt(detections._box._x)
    var y = parseInt(detections._box._y)

    if(w>h){
        x=x+(w-h)/2;
        w=h;
    }

    // draw detections into the canvas
    //faceapi.draw.drawDetections(canvas, resizedDetections)

    
    tf.engine().startScope();
    //get tensor from canvas
    var tensor = GetTensorFromCanvas(x,y,w,h)

    document.getElementById('isConvert').innerText='変換中です。もうしばらくお待ちください。';
    //Load Model
    startTime = Date.now(); // 開始時間
    model = await tf.loadLayersModel('tfmodel/model.json')
    endTime = Date.now(); // 終了時間
    console.log("Model_DL_TIME:"+(endTime - startTime)); // 何ミリ秒かかったかを表示する
    document.getElementById('isConvert').innerText='75%';
    progress_bar.setAttribute("style", "width:75%");
    progress_bar.innerText="75%";
    console.log("75%");
    //Inference
    
    console.log(tf.memory())
    console.log("InferenceStart")
    startTime = Date.now(); // 開始時間
    var prediction = await model.predict(tensor)
    endTime = Date.now(); // 終了時間
    console.log("PredictionTime:"+(endTime - startTime)); // 何ミリ秒かかったかを表示する
    tf.dispose(tensor);
    tf.dispose(model);
    document.getElementById('isConvert').innerText='変換完了';
    progress_bar.setAttribute("style", "width:90%");
    console.log("90%");
    console.log(tf.memory())

    const offset_mul = tf.scalar(127.5);
    const offset_add = tf.scalar(1.0);
    prediction=prediction.unstack(0)[0].add(offset_add).mul(offset_mul);
    offset_add.dispose();
    offset_mul.dispose();
    var image = prediction.clipByValue(0,255).toInt();
    prediction.dispose();

    
    //var temp = document.createElement("canvas");
    console.log("D");
    console.log(tf.memory())
    startTime = Date.now(); // 開始時間
    await tf.browser.toPixels(image, canvas);
    image.dispose();
    tf.engine().endScope();    
    endTime = Date.now(); // 終了時間
    console.log("TensorToCanvasTime:"+(endTime - startTime)); // 何ミリ秒かかったかを表示する
    
    
    console.log(tf.memory())
    var png = canvas.toDataURL();
    document.getElementById('translated').src=png;
    document.getElementById('isConvert').innerText='完成しました！';
    progress_bar.setAttribute("style", "width:100%");
    progress_bar.innerText="100%";

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
    //let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([128,128]).toFloat();
    let tensor = tf.browser.fromPixels(image).resizeNearestNeighbor([256,256]).toFloat();
    let offset_div = tf.scalar(127.5);
    let offset_sub = tf.scalar(1.0);
    return tensor.div(offset_div).sub(offset_sub).expandDims();
}

function downloadCanvas() {
    //let canvas = document.getElementById("canvas");
    console.log("downloadCanvas():"+canvas.width);
    let link = document.createElement("a");
    link.href = canvas.toDataURL("png");
    link.download = "Begyaru.png";
    link.click();
}



