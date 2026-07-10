// Hero Three.js scene — wobbling blobs + drifting particle field, warm palette.
// Ported from the Claude Design export (dc-runtime component) to plain JS.
// Respects prefers-reduced-motion: renders a single static frame instead of
// looping, and skips the pointermove listener entirely.

(function () {
  const canvas = document.getElementById("hero-canvas");
  if (!canvas || !window.THREE) return;

  const THREE = window.THREE;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  let W = canvas.clientWidth || (canvas.parentElement && canvas.parentElement.clientWidth) || 800;
  let H = canvas.clientHeight || 480;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(W, H, false);

  const scene = new THREE.Scene();
  const bg = new THREE.Color("#F1ECE2");
  scene.fog = new THREE.Fog(bg, 7, 22);
  const cam = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
  cam.position.set(0, 0, 12);

  scene.add(new THREE.HemisphereLight("#FFF9EE", "#E4D6BE", 1.05));
  const d1 = new THREE.DirectionalLight("#FFF3DE", 1.15);
  d1.position.set(5, 7, 9);
  scene.add(d1);
  const d2 = new THREE.DirectionalLight("#D9EFE4", 0.5);
  d2.position.set(-7, -3, 5);
  scene.add(d2);

  const cols = ["#3E9179", "#C99A54", "#C1846B", "#6FB39C", "#E1C486"];
  const blobs = [];
  for (let i = 0; i < 5; i++) {
    const g = new THREE.IcosahedronGeometry(1, 4);
    g.userData.base = Float32Array.from(g.attributes.position.array);
    const m = new THREE.MeshStandardMaterial({
      color: new THREE.Color(cols[i]),
      roughness: 0.98,
      metalness: 0.0,
      transparent: true,
      opacity: 0.9,
    });
    const mesh = new THREE.Mesh(g, m);
    const s = 1.5 + Math.random() * 1.4;
    mesh.scale.setScalar(s);
    mesh.position.set((Math.random() - 0.5) * 9, (Math.random() - 0.5) * 4.6, -2 - Math.random() * 6);
    mesh.userData = {
      ph: Math.random() * 6.28,
      sp: 0.16 + Math.random() * 0.14,
      amp: 0.10 + Math.random() * 0.07,
      rot: (Math.random() - 0.5) * 0.05,
      dx: (Math.random() - 0.5) * 0.4,
      dph: Math.random() * 6.28,
    };
    scene.add(mesh);
    blobs.push(mesh);
  }

  const pcount = 150;
  const pg = new THREE.BufferGeometry();
  const pp = new Float32Array(pcount * 3);
  for (let i = 0; i < pcount; i++) {
    pp[i * 3] = (Math.random() - 0.5) * 18;
    pp[i * 3 + 1] = (Math.random() - 0.5) * 10;
    pp[i * 3 + 2] = (Math.random() - 0.5) * 9;
  }
  pg.setAttribute("position", new THREE.BufferAttribute(pp, 3));
  const points = new THREE.Points(
    pg,
    new THREE.PointsMaterial({ color: new THREE.Color("#B5895A"), size: 0.055, transparent: true, opacity: 0.5, sizeAttenuation: true })
  );
  scene.add(points);

  const target = { x: 0, y: 0 };
  const curP = { x: 0, y: 0 };

  function onPointerMove(e) {
    const r = canvas.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) return;
    target.x = (e.clientX - r.left) / r.width - 0.5;
    target.y = (e.clientY - r.top) / r.height - 0.5;
  }
  if (!reduced) window.addEventListener("pointermove", onPointerMove);

  function resize() {
    W = canvas.clientWidth || W;
    H = canvas.clientHeight || H;
    renderer.setSize(W, H, false);
    cam.aspect = W / H;
    cam.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize);

  function wobble(mesh, t) {
    const g = mesh.geometry, arr = g.attributes.position.array, base = g.userData.base, ud = mesh.userData;
    for (let j = 0; j < arr.length; j += 3) {
      const x = base[j], y = base[j + 1], z = base[j + 2];
      const n = Math.sin(t * ud.sp + x * 2.4 + ud.ph) + Math.cos(t * ud.sp * 0.8 + y * 2.4) + Math.sin(t * ud.sp * 1.2 + z * 2.4);
      const f = 1 + ud.amp * n * 0.33;
      arr[j] = x * f; arr[j + 1] = y * f; arr[j + 2] = z * f;
    }
    g.attributes.position.needsUpdate = true;
    g.computeVertexNormals();
  }

  let raf = null;
  function render(t) {
    const tt = t * 0.001;
    curP.x += (target.x - curP.x) * 0.045;
    curP.y += (target.y - curP.y) * 0.045;
    for (const b of blobs) {
      wobble(b, tt);
      b.rotation.y += b.userData.rot;
      b.rotation.x += b.userData.rot * 0.6;
      b.position.x += Math.sin(tt * 0.2 + b.userData.dph) * 0.002 * b.userData.dx * 10;
      b.position.y += Math.cos(tt * 0.16 + b.userData.dph) * 0.0015 * 10 * 0.1;
    }
    points.rotation.y = tt * 0.02 + curP.x * 0.2;
    points.rotation.x = curP.y * 0.15;
    cam.position.x += (curP.x * 1.6 - cam.position.x) * 0.06;
    cam.position.y += (-curP.y * 1.1 - cam.position.y) * 0.06;
    cam.lookAt(0, 0, 0);
    renderer.render(scene, cam);
    if (!reduced) raf = requestAnimationFrame(render);
  }

  if (reduced) {
    for (const b of blobs) wobble(b, 2.0);
    renderer.render(scene, cam);
  } else {
    raf = requestAnimationFrame(render);
  }

  window.addEventListener("beforeunload", () => {
    if (raf) cancelAnimationFrame(raf);
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("resize", resize);
    renderer.dispose();
  });
})();
