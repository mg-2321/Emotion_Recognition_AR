import SwiftUI
import SceneKit

struct ContentView: View {
    var body: some View {
        SceneView()
            .edgesIgnoringSafeArea(.all)
    }
}

struct SceneView: UIViewRepresentable {
    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.scene = SCNScene()
        scnView.autoenablesDefaultLighting = true
        scnView.allowsCameraControl = true
        
        // Create a sphere (or any other shape you feel is calming)
        let sphereGeometry = SCNSphere(radius: 0.5)
        sphereGeometry.firstMaterial?.diffuse.contents = UIColor.systemBlue
        
        let sphereNode = SCNNode(geometry: sphereGeometry)
        sphereNode.position = SCNVector3(x: 0, y: 0, z: -2)
        
        scnView.scene?.rootNode.addChildNode(sphereNode)
        
        return scnView
    }
    
    func updateUIView(_ uiView: SCNView, context: Context) {}
}

@main
struct VRTherapyApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
