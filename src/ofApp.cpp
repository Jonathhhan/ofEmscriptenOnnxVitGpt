#include "ofApp.h"

std::string onnxModelStatus;
std::string inferenceResult;
bool onnxModelIsLoaded;

void onnx(std::string info, std::string inference) {
	if (info == "loaded") {  
		onnxModelStatus = "Onnx models loaded. Click into the window.";
		onnxModelIsLoaded = true;
		
	} else if (info == "inference") {		
		inferenceResult = inference;
	}
}

//--------------------------------------------------------------
EMSCRIPTEN_BINDINGS(Module){
	emscripten::function("onnx", &onnx);
}

//--------------------------------------------------------------
void ofApp::setup(){
	dir.listDir("images/");
	dir.allowExt("jpg");
	if( dir.size() ){
		images.assign(dir.size(), ofImage());
	}
	for(int i = 0; i < (int)dir.size(); i++){
		images[i].load(dir.getPath(i));
	}
	currentImage = 0;
	paragraph1.setWidth(900);
	paragraph1.setPosition(50, 50);
	paragraph1.setColor(ofColor(250));
	paragraph1.setBorderPadding(30);
	paragraph1.setAlignment(ofxParagraph::ALIGN_LEFT);
	paragraph1.setFont("data/font/mono.ttf", 12);
	paragraph2.setWidth(900);
	paragraph2.setPosition(50, 780);
	paragraph2.setColor(ofColor(250));
	paragraph2.setBorderPadding(30);
	paragraph2.setAlignment(ofxParagraph::ALIGN_LEFT);
	paragraph2.setFont("data/font/mono.ttf", 12);
	onnxModelIsLoaded = false;
	onnxModelStatus = "Loading Onnx models. Please wait.";
	EM_ASM(startOnnxSession());
}

//--------------------------------------------------------------
void ofApp::update(){
	paragraph1.setText(inferenceResult);
	paragraph2.setText(onnxModelStatus);
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofSetColor(255);
	images[currentImage].draw(20, 150, 960, 600);
	paragraph1.draw();
	paragraph2.draw();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {
	if (dir.size() > 0 && onnxModelIsLoaded){
		// currentImage++;
		// currentImage %= dir.size();
		currentImage = ofRandom(dir.size());
		EM_ASM(onnxInference($0, $1, $2), images[currentImage].getTexture().getTextureData().textureID, images[currentImage].getWidth(), images[currentImage].getHeight());
	}
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){

}
