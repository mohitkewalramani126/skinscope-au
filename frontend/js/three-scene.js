// Subtle decorative 3D accent for the hero header. Kept deliberately calm and
// slow-moving (per the design brief: this is a medical-adjacent tool, not a
// product landing page) — a few soft translucent forms drifting gently, not
// an attention-grabbing animation.

import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

const canvas = document.getElementById("hero-canvas");
if (canvas) {
  const hero = canvas.parentElement;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  camera.position.set(0, 0, 9);

  // Warm-toned lighting to match the cream/clay palette
  scene.add(new THREE.AmbientLight(0xf5ead9, 0.9));
  const keyLight = new THREE.DirectionalLight(0xd97757, 0.8);
  keyLight.position.set(4, 5, 6);
  scene.add(keyLight);

  const shapes = [];
  const geometries = [
    new THREE.IcosahedronGeometry(1.4, 1),
    new THREE.TorusGeometry(1.1, 0.35, 16, 64),
    new THREE.SphereGeometry(0.9, 32, 32),
  ];
  const colors = [0xbf6a4d, 0x3f7568, 0xe0c88f];

  geometries.forEach((geometry, i) => {
    const material = new THREE.MeshStandardMaterial({
      color: colors[i],
      transparent: true,
      opacity: 0.35,
      roughness: 0.5,
      metalness: 0.1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set((i - 1) * 3.2, Math.sin(i) * 0.6, -i * 1.5);
    mesh.userData.driftSpeed = 0.15 + i * 0.05;
    scene.add(mesh);
    shapes.push(mesh);
  });

  function resize() {
    const width = hero.clientWidth;
    const height = hero.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);
  resize();

  let mouseX = 0;
  let mouseY = 0;
  window.addEventListener("mousemove", (e) => {
    mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  const clock = new THREE.Clock();

  function animate() {
    const t = clock.getElapsedTime();

    shapes.forEach((mesh) => {
      mesh.rotation.x = t * 0.08 * mesh.userData.driftSpeed;
      mesh.rotation.y = t * 0.12 * mesh.userData.driftSpeed;
    });

    // gentle parallax, not a full camera swing
    camera.position.x += (mouseX * 1.2 - camera.position.x) * 0.02;
    camera.position.y += (-mouseY * 0.6 - camera.position.y) * 0.02;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  // Respect reduced-motion preferences -- render one static frame instead of looping
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReducedMotion) {
    renderer.render(scene, camera);
  } else {
    animate();
  }
}
