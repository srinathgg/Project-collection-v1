// Source - https://stackoverflow.com/q/41089475
// Posted by user5955461, modified by community. See post 'Timeline' for change history
// Retrieved 2026-02-23, License - CC BY-SA 4.0

var renderer = new THREE.WebGLRenderer();

// I have attached the three.js library in the script tag. I don't know what seems to be problem.

var scene = new THREE.Scene();

var camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set = (0, 0, 10);
camera.lookAt(camera.position);
scene.add(camera);

var geometry = new THREE.Geometry();
geometry.vertices.push(new THREE.Vector3(0.0, 1.0, 0.0));
geometry.vertices.push(new THREE.Vector3(-1.0, -1.0, 0.0));
geometry.vertices.push(new THREE.Vector3(1.0, -1.0, 0.0));
geometry.faces.push(new THREE.Face3(0, 1, 2));

var material = new THREE.BasicMeshMaterial({
    color: 0xFFFFFF,
    side: THREE.DoubleSide
});

var mesh = new THREE.Mesh(geometry, material);
mesh.position.set(-1.5, 0.0, 4.0);
scene.add(mesh);

function render() {
    renderer.render(scene, camera);
}

render();
