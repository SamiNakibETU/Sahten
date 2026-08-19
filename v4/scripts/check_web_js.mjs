// Vérifie la syntaxe de TOUT le JavaScript servi : fichiers .js (modules ES)
// et scripts embarqués dans les pages HTML.
// Raison d'être : deux pannes en production causées par une erreur de syntaxe
// introduite en éditant un fichier via un script (regex coupée par un saut de
// ligne, apostrophe non échappée). `node --check` seul ne suffit pas — il
// analyse en CommonJS et ignore le HTML.
import { readFileSync, readdirSync, writeFileSync, mkdtempSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';

const racine = 'web_static';
const tmp = mkdtempSync(join(tmpdir(), 'jscheck-'));
let echecs = 0;

const verifier = (nom, source, module) => {
  const f = join(tmp, 'x.' + (module ? 'mjs' : 'js'));
  writeFileSync(f, source);
  try {
    execFileSync(process.execPath, ['--check', f], { stdio: 'pipe' });
    console.log('  OK   ' + nom);
  } catch (e) {
    echecs++;
    const msg = (e.stderr?.toString() || e.message).split('\n').slice(0, 4).join('\n      ');
    console.log('  ÉCHEC ' + nom + '\n      ' + msg);
  }
};

for (const f of readdirSync(join(racine, 'js'))) {
  if (f.endsWith('.js')) {
    const src = readFileSync(join(racine, 'js', f), 'utf8');
    verifier('js/' + f, src, src.includes('export ') || src.includes('import '));
  }
}
for (const f of readdirSync(racine)) {
  if (!f.endsWith('.html')) continue;
  const html = readFileSync(join(racine, f), 'utf8');
  const blocs = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)];
  blocs.forEach((m, i) => {
    const estModule = /type=["']module["']/.test(m[0]);
    verifier(`${f} [script ${i + 1}/${blocs.length}]`, m[1], estModule);
  });
}
if (echecs) { console.log(`\n${echecs} script(s) invalide(s) — NE PAS DÉPLOYER`); process.exit(1); }
console.log('\nTous les scripts servis sont syntaxiquement valides.');
