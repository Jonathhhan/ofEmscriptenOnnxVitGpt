var onnxEncoderSession, onnxDecoderSession;
var calls = 0;

startOnnxSession = async function() {
  	onnxEncoderSession = await ort.InferenceSession.create("encoder_model_quantized.onnx", {executionProviders: ["wasm"] });
  	onnxDecoderSession = await ort.InferenceSession.create("decoder_model_quantized.onnx", {executionProviders: ["wasm"] });
	Module.onnx("loaded", "");
}

onnxInference = async function(textureID, textureWidth, textureHeight) {
	const rgbData = await getRgbData(textureID, textureWidth, textureHeight);
	const feeds = {'pixel_values': new ort.Tensor('float32', rgbData, [1,3,224,224])};
	const imageResults = await onnxEncoderSession.run(feeds);
	gptLoop(++calls, imageResults);
}

async function gptLoop(id, imageResults) {
	Module.onnx("inference", "");
	var textOutput = "";
	var generatedToken = 0;
	var bigInt64Array = [];
	bigInt64Array[0] = 0n;
	while (generatedToken < 32) {
		await new Promise(resolve => setTimeout(resolve, 0));
		if (id !== calls) break;
		generatedToken++;
		bigInt64Array = bigInt64Array.slice(Math.max(bigInt64Array.length - 50, 0));
		const inputTensor = new ort.Tensor("int64", bigInt64Array, [1, bigInt64Array.length]);
		const feeds = { input_ids: inputTensor, encoder_hidden_states: imageResults["last_hidden_state"] };
		const results = await onnxDecoderSession.run(feeds);
		const outputData = results["logits"].data.slice(50257 * (bigInt64Array.length - 1), 50257 + 50257 * (bigInt64Array.length - 1));
		const entries = Object.entries(outputData);
		const sorted = entries.sort((a, b) => b[1] - a[1]);
		const newToken = parseInt(sorted[0][0]);
		bigInt64Array.push(BigInt(newToken));
		var newWord = GPTTokenizer_p50k_edit.decode([newToken]);
		newWord = newWord.replace(/(\r\n|\n|\r)/gm, " ");
		if (newWord != "<|endoftext|>") {
			textOutput = textOutput + newWord;
			const chars = '].!;?)`';
			const lastChar = textOutput.charAt(textOutput.length - 1);
			Module.onnx("inference", textOutput);
			console.log("Loop number:", calls, "Token number:", generatedToken, "Generated text:", textOutput);
		} else {
			console.log("End of text!");
			break;
		}
	}
}

async function getRgbData(textureID, textureWidth, textureHeight) {
	var fb1 = GLctx.createFramebuffer();
	const w = textureWidth, h = textureHeight;
	const texture = GL.textures[textureID];
	GLctx.bindFramebuffer(GLctx.FRAMEBUFFER, fb1);
	GLctx.framebufferTexture2D(GLctx.FRAMEBUFFER, GLctx.COLOR_ATTACHMENT0, GLctx.TEXTURE_2D, texture, 0);
	GLctx.bindFramebuffer(GLctx.FRAMEBUFFER, null);
	GLctx.bindFramebuffer(GLctx.FRAMEBUFFER, fb1);
	const data1 = new Uint8Array(w * h * 4);
	const imageData1 = new ImageData(new Uint8ClampedArray(data1.buffer), w, h);
	GLctx.readPixels(0, 0, w, h, GLctx.RGBA, GLctx.UNSIGNED_BYTE, data1);
	GLctx.bindFramebuffer(GLctx.FRAMEBUFFER, null);
	const canvas = new OffscreenCanvas(224, 224);
	const ctx = canvas.getContext("2d");
	ctx.putImageData(imageData1, 0, 0);
	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
	var rgbData = [[], [], []]; // [r, g, b]
	// remove alpha and put into correct shape:
        for(var i = 0; i < imageData.data.length; i += 4) { 
		var x = (i/4) % canvas.width;
		var y = Math.floor((i/4) / canvas.width)
		if(!rgbData[0][y]) rgbData[0][y] = [];
		if(!rgbData[1][y]) rgbData[1][y] = [];
		if(!rgbData[2][y]) rgbData[2][y] = [];
		rgbData[0][y][x] = d[i+0]/255;
		rgbData[1][y][x] = d[i+1]/255;
		rgbData[2][y][x] = d[i+2]/255;
		// From CLIP repo: Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
		rgbData[0][y][x] = (rgbData[0][y][x] - 0.48145466) / 0.26862954;
		rgbData[1][y][x] = (rgbData[1][y][x] - 0.4578275) / 0.26130258;
		rgbData[2][y][x] = (rgbData[2][y][x] - 0.40821073) / 0.27577711;
	}
	rgbData = Float32Array.from(rgbData.flat().flat());
	return rgbData;
}
