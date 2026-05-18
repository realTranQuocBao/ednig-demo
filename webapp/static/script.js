// EDNIG web UI logic: drag-drop, upload, before/after slider, download.
(function () {
  const dropZone   = document.getElementById('drop-zone');
  const pickBtn    = document.getElementById('pick-btn');
  const fileInput  = document.getElementById('file-input');
  const resultSec  = document.getElementById('result-section');
  const loading    = document.getElementById('loading');
  const errorBox   = document.getElementById('error-box');
  const imgBefore  = document.getElementById('img-before');
  const imgAfter   = document.getElementById('img-after');
  const beforeWrap = document.getElementById('before-wrap');
  const slider     = document.getElementById('slider');
  const handle     = document.getElementById('handle');
  const metaSize   = document.getElementById('meta-size');
  const metaMode   = document.getElementById('meta-mode');
  const metaTime   = document.getElementById('meta-time');
  const dlPng      = document.getElementById('download-png');
  const dlJpg      = document.getElementById('download-jpg');
  const resetBtn   = document.getElementById('reset-btn');

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
  }
  function hideError() {
    errorBox.classList.add('hidden');
    errorBox.textContent = '';
  }
  function setLoading(on) {
    if (on) loading.classList.remove('hidden'); else loading.classList.add('hidden');
  }
  function showResult(on) {
    resultSec.classList.toggle('hidden', !on);
  }

  function setSliderPos(pct) {
    const p = Math.max(0, Math.min(100, pct));
    beforeWrap.style.width = p + '%';
    handle.style.left = p + '%';
    slider.value = p;
    // Keep the "before" image aligned with the right edge of its container
    imgBefore.style.width = (10000 / p) + '%';
    if (!isFinite(parseFloat(imgBefore.style.width))) {
      imgBefore.style.width = '200%';
    }
  }

  slider.addEventListener('input', (e) => setSliderPos(parseFloat(e.target.value)));

  // ---- Drag and drop ----
  ['dragenter', 'dragover'].forEach(ev => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(ev => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault(); e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });
  dropZone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
  dropZone.addEventListener('click', (e) => {
    // Avoid double-click when the inner button is hit
    if (e.target === pickBtn) return;
    fileInput.click();
  });
  pickBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (f) handleFile(f);
    fileInput.value = '';
  });

  resetBtn.addEventListener('click', () => {
    showResult(false); hideError();
  });

  // ---- Upload + render ----
  async function handleFile(file) {
    hideError();
    if (!file.type.startsWith('image/')) {
      showError('Vui lòng chọn file ảnh.');
      return;
    }
    const fd = new FormData();
    fd.append('image', file);

    setLoading(true);
    try {
      const resp = await fetch('/api/enhance', { method: 'POST', body: fd });
      const data = await resp.json();
      if (!data.ok) {
        showError(data.error || 'Có lỗi xảy ra.');
        return;
      }
      // Render
      imgBefore.src = data.original_b64;
      imgAfter.src  = data.enhanced_b64;
      setSliderPos(50);
      metaSize.textContent = data.width + ' × ' + data.height + ' px';
      metaMode.textContent = (data.mode === 'model') ? 'EDNIG model' : 'Classical fallback';
      metaTime.textContent = data.ms + ' ms';
      dlPng.href = '/api/download?session=' + encodeURIComponent(data.session) + '&kind=enhanced&format=png';
      dlJpg.href = '/api/download?session=' + encodeURIComponent(data.session) + '&kind=enhanced&format=jpg';
      showResult(true);
      resultSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      showError('Lỗi mạng: ' + err.message);
    } finally {
      setLoading(false);
    }
  }

  // Init slider position
  setSliderPos(50);
})();
