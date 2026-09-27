/* Dashboard interactions.
 *
 * Four, each answering a question a reviewer actually asks of a long
 * recording. Nothing here is decoration: a chart that moves without telling
 * you something new is worse than a still one.
 *
 *   1. Theme        - the report is read on other people's screens.
 *   2. Sensor tabs  - "does the other sensor agree?"
 *   3. Linked zoom  - "what else was happening at that minute?"
 *   4. Bin inspect  - "that spike in the histogram: were those real beats?"
 */
(function () {
  "use strict";

  var PALETTE = {
    light: {ink:'#111820', muted:'#5a6572', grid:'rgba(120,132,148,.20)',
            trace:'#2c7c9c', event:'#b3283c', good:'#3f8a63'},
    dark:  {ink:'#e8ecf1', muted:'#9aa5b2', grid:'rgba(150,162,178,.18)',
            trace:'#5fb3d4', event:'#e8697c', good:'#68bd92'}
  };
  /* Placeholders baked into the serialised figures, swapped at view time. */
  var SWAP = {
    '#8a94a3':'muted', 'rgba(138,148,163,0.22)':'grid',
    '#3d8fb0':'trace', '#d1495b':'event', '#4f9e72':'good'
  };

  var DATA = window.__ECG__ || {};
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };

  function effectiveTheme() {
    var set = document.documentElement.getAttribute('data-theme');
    if (set === 'dark' || set === 'light') return set;
    return window.matchMedia &&
           window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  /* ---------------------------------------------------------- 1. theme */

  function recolour(node, pal) {
    if (Array.isArray(node)) { node.forEach(function (n) { recolour(n, pal); }); return; }
    if (!node || typeof node !== 'object') return;
    Object.keys(node).forEach(function (k) {
      var v = node[k];
      if (typeof v === 'string' && SWAP[v]) node[k] = pal[SWAP[v]];
      else if (v && typeof v === 'object') recolour(v, pal);
    });
  }

  function applyTheme() {
    if (!window.Plotly) return;
    var pal = PALETTE[effectiveTheme()];
    document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      try {
        recolour(gd.layout, pal);
        recolour(gd.data, pal);
        if (gd.layout.font) gd.layout.font.color = pal.muted;
        (gd.layout.annotations || []).forEach(function (a) {
          if (a.font) a.font.color = pal.muted;
        });
        Plotly.react(gd, gd.data, gd.layout, gd._context);
      } catch (e) { /* one bad plot must not blank the page */ }
    });
  }

  function initTheme() {
    var saved = store.get('ecg-theme');
    if (saved === 'dark' || saved === 'light') {
      document.documentElement.setAttribute('data-theme', saved);
    }
    var btn = document.getElementById('theme-btn');
    if (!btn) return;
    function label() {
      btn.textContent = effectiveTheme() === 'dark' ? 'Light' : 'Dark';
    }
    label();
    btn.addEventListener('click', function () {
      var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      store.set('ecg-theme', next);
      label();
      applyTheme();
    });
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: dark)');
      var onChange = function () { label(); applyTheme(); };
      if (mq.addEventListener) mq.addEventListener('change', onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }
  }

  /* ----------------------------------------------------- 2. sensor tabs */

  function initTabs() {
    var tabs = document.querySelectorAll('[data-sensor-tab]');
    if (!tabs.length) return;
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        var id = tab.getAttribute('data-sensor-tab');
        tabs.forEach(function (t) {
          t.classList.toggle('on', t === tab);
          t.setAttribute('aria-selected', String(t === tab));
        });
        document.querySelectorAll('[data-sensor-panel]').forEach(function (p) {
          p.hidden = p.getAttribute('data-sensor-panel') !== id;
        });
        /* A plot sized while its container was hidden lays out at zero width
           and stays that way; it has to be resized once it is on screen. */
        if (window.Plotly) {
          document.querySelectorAll('[data-sensor-panel]:not([hidden]) .js-plotly-plot')
            .forEach(function (gd) { try { Plotly.Plots.resize(gd); } catch (e) {} });
        }
      });
    });
  }

  /* ----------------------------------------------------- 3. linked zoom */

  function initLinkedZoom() {
    var syncing = false;
    document.querySelectorAll('[data-link-time]').forEach(function (gd) {
      if (!gd.on) return;
      gd.on('plotly_relayout', function (ev) {
        if (syncing) return;
        var group = gd.getAttribute('data-link-time');
        /* Plotly reports a new range two different ways: as
           'xaxis.range[0]' / '[1]' when the user drags, and as an
           'xaxis.range' array when it is set in code. Handling only the
           first means zoom syncs on a drag but silently does nothing
           otherwise. */
        var range;
        if (ev['xaxis.autorange']) range = null;
        else if (ev['xaxis.range[0]'] !== undefined)
          range = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
        else if (Array.isArray(ev['xaxis.range']))
          range = ev['xaxis.range'].slice();
        else return;

        syncing = true;
        document.querySelectorAll('[data-link-time="' + group + '"]')
          .forEach(function (other) {
            if (other === gd) return;
            try {
              Plotly.relayout(other, range ? {'xaxis.range': range}
                                           : {'xaxis.autorange': true});
            } catch (e) {}
          });
        syncing = false;

        var note = document.getElementById('zoom-note-' + group);
        if (note) {
          note.textContent = range
            ? 'Showing ' + range[0].toFixed(1) + '-' + range[1].toFixed(1) +
              ' min. Double-click a chart to reset.'
            : '';
        }
      });
    });
  }

  /* ---------------------------------------------------- 4. bin inspector */

  function initInspector() {
    document.querySelectorAll('[data-events-for]').forEach(function (gd) {
      if (!gd.on) return;
      var sensor = gd.getAttribute('data-events-for');
      gd.on('plotly_click', function (ev) {
        var pt = ev.points && ev.points[0];
        if (!pt) return;
        showBin(sensor, pt.pointIndex, pt.y);
      });
    });
  }

  function showBin(sensor, index, count) {
    var host = document.getElementById('inspect-' + sensor);
    var cap = document.getElementById('inspect-cap-' + sensor);
    if (!host) return;

    var bins = (DATA.examples || {})[sensor] || [];
    var beats = bins[index] || [];
    var pal = PALETTE[effectiveTheme()];

    if (!beats.length) {
      if (cap) {
        cap.textContent = count
          ? 'No stored waveforms for that block.'
          : 'That block holds no flagged beats.';
      }
      try { Plotly.purge(host); } catch (e) {}
      host.innerHTML = '';
      return;
    }

    var fs = DATA.fs || 360;
    var traces = beats.map(function (b) {
      var n = b.y.length;
      var x = new Array(n);
      for (var i = 0; i < n; i++) x[i] = (i - n / 2) / fs;
      return {
        x: x, y: b.y, mode: 'lines', type: 'scatter',
        line: {width: 1.4, color: pal.event}, opacity: 0.75,
        name: b.t.toFixed(1) + ' s',
        hovertemplate: 'beat at %{fullData.name}<extra></extra>'
      };
    });

    Plotly.react(host, traces, {
      height: 230,
      margin: {l: 58, r: 16, t: 10, b: 42},
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: {color: pal.muted, size: 11.5},
      xaxis: {title: {text: 'seconds from the beat', font: {size: 11}},
              gridcolor: pal.grid, zeroline: false, linecolor: pal.grid},
      yaxis: {title: {text: 'mV', font: {size: 11}},
              gridcolor: pal.grid, zeroline: false, linecolor: pal.grid},
      showlegend: false,
      shapes: [{type: 'line', x0: 0, x1: 0, yref: 'paper', y0: 0, y1: 1,
                line: {color: pal.muted, width: 1, dash: 'dot'}}]
    }, {displaylogo: false, responsive: true});

    if (cap) {
      /* The click payload does not always carry the bar's height; say what is
         known rather than printing "undefined" at the reader. */
      var total = (typeof count === 'number' && isFinite(count)) ? count : null;
      var of = total !== null ? ' of ' + total : '';
      cap.textContent = 'Showing ' + beats.length + of + ' flagged beat' +
        (beats.length === 1 && total === null ? '' : 's') +
        ' from that block, around ' + Math.round(beats[0].t) +
        ' s. Beats that look unlike the others are candidate false positives.';
    }
  }

  /* ------------------------------------------------------------- boot */

  function boot() {
    initTheme();
    applyTheme();
    initTabs();
    initLinkedZoom();
    initInspector();

    /* Open each sensor's inspector on its busiest block, so the panel shows
       something real instead of an empty frame waiting to be clicked. */
    Object.keys(DATA.examples || {}).forEach(function (sensor) {
      var bins = DATA.examples[sensor] || [];
      var counts = (DATA.counts || {})[sensor] || [];
      var best = -1, bestN = 0;
      bins.forEach(function (b, i) {
        if (b.length && (counts[i] || 0) >= bestN) { best = i; bestN = counts[i] || 0; }
      });
      if (best >= 0) showBin(sensor, best, bestN);
    });
  }

  if (document.readyState === 'complete') boot();
  else window.addEventListener('load', boot);
})();
