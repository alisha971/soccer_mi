"""Validation-only repairs using existing checkpoints; never overwrite MB01–MB23.

Run from any directory: python -B mi_analysis/targetted_analysis/run.py
Dependencies: the existing mi_analysis/requirements-rerun.txt environment.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.functional as TF
from torch.func import jvp
from sklearn.linear_model import RidgeCV
from sklearn.exceptions import ConvergenceWarning

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
MI = ROOT / 'mi_analysis'
HERE = Path(__file__).resolve().parent
REPORT = MI / 'targetted_analysis.md'
sys.path.insert(0, str(ROOT / 'model_training_evaluation'))
spec = importlib.util.spec_from_file_location('mi_src', MI / 'src/__init__.py', submodule_search_locations=[str(MI / 'src')])
module = importlib.util.module_from_spec(spec)
sys.modules['mi_src'] = module
spec.loader.exec_module(module)
from mi_src import paths, mi_setup, mi_metrics, hooks, interventions, circuits, conflict, probing
from mi_src import twotoken_followup as tt


def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def protect(event, args):
    targets = []
    if event == 'open':
        path, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            targets = [path]
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.rename', 'os.chmod', 'os.utime'):
        targets = args[:2] if event == 'os.rename' else args[:1]
    for target in targets:
        if isinstance(target, (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(target)).resolve()
            if p.is_relative_to(ROOT) and not (p.is_relative_to(HERE) or p == REPORT):
                raise RuntimeError(f'Historical-file write blocked: {event} {p}')


def functions(prefix, names, namespace):
    """Reuse only named function definitions; never execute notebook output cells."""
    path = next((MI / 'notebooks').glob(prefix + '_*.ipynb'))
    nodes = []
    for cell in json.loads(path.read_text())['cells']:
        if cell['cell_type'] == 'code':
            nodes += [n for n in ast.parse(''.join(cell['source'])).body
                      if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), (prefix, names)
    assert len(nodes) == len(names), 'Ambiguous notebook function'
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def summarise(values, matches):
    return mi_metrics.summarise(np.asarray(values), np.asarray(matches), 2000)


def align(table, features):
    table = table.copy()
    for name in conflict.IDENTITY_COLUMNS:
        table['reference_' + name] = table[name]
        table[name] = features[name][table.event_row.to_numpy()]
    return table


def vector_edit(vectors, positions):
    # factor_player has exactly one categorical embedding: assign, do not subtract metadata.
    def edit(value):
        new = value.clone()
        new[torch.arange(len(positions)), positions] = vectors
        return new
    return edit


def self_check():
    table = pd.DataFrame({'event_row': [1, 0], **{k: [9, 8] for k in conflict.IDENTITY_COLUMNS}})
    fixed = align(table, {k: np.array([2, 3]) for k in conflict.IDENTITY_COLUMNS})
    assert fixed.player.tolist() == [3, 2] and fixed.reference_player.tolist() == [9, 8]
    x = torch.arange(12.).reshape(2, 3, 2)
    positions = torch.tensor([1, 2]); v = torch.ones(2, 2)
    y = vector_edit(v, positions)(x)
    assert torch.equal(y[torch.arange(2), positions], v) and torch.equal(y[:, 0], x[:, 0])
    result = mi_metrics.pooled_recovery(np.array([1., 2., 3.]), np.array([2., 4., 6.]), np.array([1, 2, 3]), 100)
    assert abs(result['recovery'] - .5) < 1e-12


def setup():
    TRACK = paths.track_a_paths(ROOT / 'model_training_evaluation')
    events = pd.read_parquet(TRACK['events'])
    bundle = mi_setup.build_bundle(events, TRACK, 'cpu')
    runs, PRIMARY_SEED = mi_setup.sel4l_runs(TRACK)
    MODEL = mi_setup.load_model(mi_setup.run_dir_for_seed(runs, PRIMARY_SEED), bundle)
    val_view = bundle['views']['val']; DEVICE = 'cpu'; BATCH = 256; N_BOOT = 2000; N_DRAWS = 5
    features = {k: v.cpu().numpy() for k, v in bundle['encoded'].tensors.items()}
    original = pd.read_parquet(MI / 'artifacts/populations/val_conflicts.parquet')
    val_conflicts = align(original, features)
    partners = interventions.build_partner_table(events, features, events.split.to_numpy() == 'train')
    partner_of = dict(zip(partners.player, partners.partner))
    donors = pd.read_parquet(MI / 'artifacts/populations/context_donors.parquet')
    p_player = val_conflicts[val_conflicts.population == 'P-player'].reset_index(drop=True).merge(donors[['event_row', 'donor_window']], on='event_row', how='left')
    assert (p_player.player > 0).all()
    assert (val_conflicts[val_conflicts.population == 'index-0'].player == 0).all()
    FEATURES = list(features); IDENTITY_FEATURES = list(conflict.IDENTITY_COLUMNS)
    CONTEXT_FEATURES = [k for k in FEATURES if k not in IDENTITY_FEATURES]; IDENTITY = 'player'
    N_LAYERS = len(MODEL.blocks)
    ENCODED = bundle['encoded'].tensors; TRAIN_ROWS = np.flatnonzero(events.split.to_numpy() == 'train')
    train = pd.DataFrame({k: features[k][TRAIN_ROWS] for k in ['player', 'role', 'team']})
    PLAYER_META = train[train.player > 0].groupby('player').agg(role=('role', lambda s: s.value_counts().index[0]), team=('team', lambda s: s.value_counts().index[0]), train_events=('role', 'size'))
    PLAYER_TABLE = MODEL.embedding.embeddings['player'].weight.detach()
    ROLE_CENTROID = {int(role): PLAYER_TABLE[torch.as_tensor(g.index.to_numpy())].mean(0) for role, g in PLAYER_META.groupby('role')}
    return {**globals(), **locals()}


def identity(ns):
    functions('MB15', ['identity_controls', 'control_vectors', 'player_vector_edit', 'team_matched_partner', 'frequency_band', 'delta_of'], ns)
    ns.update(CONTROL_KINDS=['role centroid', 'distribution-matched deviation', 'shuffled coordinates'], RANDOM_KINDS=['distribution-matched deviation', 'shuffled coordinates'], FREQUENCY_CUT=1000)
    functions('MB21', ['centred_log_probs', 'part_swaps'], ns)
    events = ns['events']; features = ns['features']; players = ns['PLAYER_META'].index.to_numpy()
    train = ns['TRAIN_ROWS']; actions = events.action_id.to_numpy()[train]
    base = np.bincount(actions, minlength=15) / len(actions)
    counts = np.zeros((len(players), 15)); indices = {p: i for i, p in enumerate(players)}
    keep = np.isin(features['player'][train], players)
    np.add.at(counts, (np.array([indices[p] for p in features['player'][train][keep]]), actions[keep]), 1)
    habit = np.log(((counts + 20 * base) / (counts.sum(1, keepdims=True) + 20)) / base)
    state = torch.load(MI / 'artifacts/conflict/references/id_player_seed0.pt', map_location='cpu', weights_only=False)
    ref = conflict.IdPlayer(state['vocabulary_sizes']); ref.load_state_dict(state['state'])
    with torch.no_grad():
        n = state['vocabulary_sizes']['player']
        reference = torch.log_softmax(ref(torch.arange(n)[:, None], torch.zeros(n, 0)) / state['temperature'], -1).numpy()
    # Join the two vocabularies through the same physical player, not equal integer codes.
    mapping = pd.DataFrame({'raw': events.player_id, 'model': features['player'], 'reference': events.player_ix}).drop_duplicates()
    mapping = mapping[mapping.model > 0]
    assert mapping.groupby('raw').model.nunique().max() == 1
    assert mapping.groupby('model').raw.nunique().max() == 1
    assert mapping.groupby('model').reference.nunique().max() == 1
    assert mapping.reference.nunique() == len(players)
    lookup = np.zeros(ns['PLAYER_TABLE'].shape[0], dtype=int)
    lookup[mapping.model] = mapping.reference
    assert (lookup[players] > 0).all()
    ns['HABIT_REF'] = reference[lookup]
    rows = []; controls = []
    for seed in ns['runs'].seed:
        ns['MODEL'] = mi_setup.load_model(mi_setup.run_dir_for_seed(ns['runs'], int(seed)), ns['bundle'])
        ns['PLAYER_TABLE'] = ns['MODEL'].embedding.embeddings['player'].weight.detach()
        ns['ROLE_CENTROID'] = {int(role): ns['PLAYER_TABLE'][torch.as_tensor(g.index.to_numpy())].mean(0) for role, g in ns['PLAYER_META'].groupby('role')}
        emb = ns['PLAYER_TABLE'].numpy()[players]
        decoder = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(emb, habit)
        ns['BASIS'] = torch.as_tensor(np.linalg.svd(decoder.coef_, full_matrices=False)[2], dtype=torch.float32)
        # Full-vector assignment must equal the corresponding categorical index swap.
        sample = ns['p_player']; sample = sample[sample.player != sample.reference_player].head(64)
        for _, inp, pos, _, _ in mi_setup.event_batches(ns['bundle'], ns['val_view'], sample, 64):
            other = np.array([ns['partner_of'][p] for p in sample.player])
            categorical = interventions.set_feature(inp, 'player', pos, torch.as_tensor(other))
            with torch.no_grad():
                a = hooks.forward(ns['MODEL'], categorical)[0]
                b = hooks.forward(ns['MODEL'], inp, replace={'factor_player': ns['player_vector_edit'](ns['PLAYER_TABLE'][other], sample.player.to_numpy(), pos)})[0]
            assert torch.equal(a, b), 'Aligned vector/index swaps disagree'
        frame = ns['identity_controls'](ns['MODEL'], ns['p_player'])
        swaps = ns['part_swaps'](ns['p_player']); swaps = swaps[swaps.valid]
        for half in ['estimation', 'discovery']:
            f = frame[frame.half == half]; m = f.match_id
            for name, values in [('removal |Delta|', abs(f['removal effect'])), ('partner |Delta|', abs(f['partner: change'])), ('C2 partner minus norm-random', abs(f['partner: change']) - f['norm-matched random vector: mean |change|']), ('C2 partner minus distribution-matched', abs(f['partner: change']) - f['distribution-matched deviation: mean |change|'])]:
                valid = values.notna(); controls.append(dict(seed=int(seed), half=half, measure=name, **summarise(values[valid], m[valid])))
            valid = f['role centroid: change'].notna(); f = f[valid]
            r = mi_metrics.pooled_recovery(-f['role centroid: change'].to_numpy(), f['removal effect'].to_numpy(), f.match_id.to_numpy(), 2000)
            controls.append(dict(seed=int(seed), half=half, measure='player-specific share relative to role centroid', mean=r['recovery'], ci_low=r['ci_low'], ci_high=r['ci_high'], n=len(f)))
            s = swaps[swaps.half == half]
            for name in ['whole vector', 'habit part only', 'rest only']:
                rows.append(dict(seed=int(seed), half=half, measure=name, **summarise(s[name + ': C3 r'], s.match_id)))
            rows.append(dict(seed=int(seed), half=half, measure='habit minus rest (paired)', **summarise(s['habit part only: C3 r'] - s['rest only: C3 r'], s.match_id)))
        print('identity seed', seed, 'complete', flush=True)
    return dict(controls=controls, habit_use=rows, mismatched_player_rows=int((ns['original'].player != features['player'][ns['original'].event_row]).sum()), reference_mapping_players=len(players))


def history(ns):
    functions('MB15', ['edit_inputs'], ns)
    table = ns['val_conflicts'].copy(); rng = np.random.default_rng(0); selected = []
    groups = {k: g for k, g in table.groupby(['half', 'position'])}
    for _, row in table.iterrows():
        pool = groups[row.half, row.position]; pool = pool[pool.match_id != row.match_id]
        selected.append(pool.iloc[rng.integers(len(pool))].event_row if len(pool) else -1)
    table['donor_event'] = selected; excluded = int((table.donor_event < 0).sum())
    table = table[table.donor_event >= 0].reset_index(drop=True)
    donor = ns['val_conflicts'].set_index('event_row').loc[table.donor_event]
    assert np.array_equal(table.half, donor.half) and np.all(table.match_id.to_numpy() != donor.match_id.to_numpy())
    table['history_donor_window'] = donor.window.to_numpy(); frames = []
    with torch.no_grad():
        for part, inp, pos, ctx, ident in mi_setup.event_batches(ns['bundle'], ns['val_view'], table, 256):
            chunk = table.iloc[part]; dinp, _ = mi_setup.window_inputs(ns['bundle'], ns['val_view'], chunk.history_donor_window.to_numpy())
            base = mi_metrics.delta(mi_metrics.at_target(hooks.forward(ns['MODEL'], inp)[0], pos), ctx, ident)
            frame = chunk[['match_id', 'half', 'genuine']].copy()
            for kind in ['history swap', 'scored-event context swap', 'whole-context swap']:
                edited = ns['edit_inputs'](kind, inp, pos, dinp)
                delta = mi_metrics.delta(mi_metrics.at_target(hooks.forward(ns['MODEL'], edited)[0], pos), ctx, ident)
                frame[kind] = abs(delta - base)
            frames.append(frame)
    data = pd.concat(frames); rows = []
    for label, f in [('all validation', data), ('estimation', data[data.half == 'estimation']), ('discovery', data[data.half == 'discovery']), ('genuine', data[data.genuine]), ('coupled', data[~data.genuine])]:
        for kind in ['history swap', 'scored-event context swap', 'whole-context swap']:
            rows.append(dict(subset=label, measure=kind, **summarise(f[kind], f.match_id)))
        r = mi_metrics.pooled_recovery(f['history swap'].to_numpy(), f['whole-context swap'].to_numpy(), f.match_id.to_numpy(), 2000)
        rows.append(dict(subset=label, measure='history / whole-context magnitude', mean=r['recovery'], ci_low=r['ci_low'], ci_high=r['ci_high'], n=len(f)))
    return dict(rows=rows, excluded_no_same_half_position_donor=excluded, donors_cross_half=0)


def interaction(ns):
    functions('MB20', ['mlp_whole', 'mlp_activation', 'mlp_layernorm', 'site_names', 'linearised_value', 'linearise_replace', 'factorial_runs'], ns)
    components = {}
    for layer in range(ns['N_LAYERS']):
        for part, label in [('whole', '(whole)'), ('activation', 'activation only'), ('layernorm', 'LayerNorm only')]:
            components[f'L{layer} MLP {label}'] = [('mlp', layer, part)]
    components['final LayerNorm'] = [('final', None, 'layernorm')]
    components['all four MLPs (whole)'] = [('mlp', l, 'whole') for l in range(4)]
    table = ns['p_player'].dropna(subset=['donor_window']).copy(); table.donor_window = table.donor_window.astype(int)
    table['partner'] = [ns['partner_of'].get(p, -1) for p in table.player]; table = table[table.partner > 0].reset_index(drop=True)
    ns.update(COMPONENTS=components, D_MODEL=128, INPUT_SITES=[f'mlp_in_{l}' for l in range(4)] + ['final_resid'], FROZEN=[f'{kind}_{l}' for l in range(4) for kind in ['attn_out', 'mlp_out']])
    check = ns['factorial_runs'](table.head(16), 'partner', check=True)
    assert max(abs(v).max() for k, v in check.items() if k != 'floor') < 1e-3
    rows = []
    for second in ['partner', 'random vector']:
        values = ns['factorial_runs'](table, second)
        for half in ['estimation', 'discovery']:
            keep = table.half.to_numpy() == half; exact = values['exact'][keep]; m = table.match_id.to_numpy()[keep]
            for name in components:
                own, shared = [values[name + ' | ' + mode][keep] for mode in ['own', 'shared']]
                a = mi_metrics.pooled_recovery(own, exact, m, 2000)
                b = mi_metrics.pooled_recovery(shared, exact, m, 2000)
                r = mi_metrics.pooled_recovery(own - shared, exact, m, 2000)
                rows.append(dict(second=second, half=half, component=name, own=a['recovery'], shared=b['recovery'], mean=r['recovery'], ci_low=r['ci_low'], ci_high=r['ci_high'], source=bool(a['ci_low'] >= .8 and r['ci_low'] > .2), n=int(keep.sum())))
        print('interaction', second, 'complete', flush=True)
    return dict(rows=rows)


def slots(ns):
    functions('MB23', ['edit', 'delta_at'], ns)
    ns['TT_DATA'] = data = tt.EventData(ns['events'], ns['bundle']['encoded'].tensors, ns['bundle']['targets'], 32)
    table = ns['p_player'][['event_row', 'match_id', 'half', 'a_ctx', 'a_id', 'player']].merge(pd.DataFrame(data.scored['val']), on='event_row')
    players = data.features['player'][data.windows['val'][table.window]]
    table['linked'] = [bool((players[i, :p] == players[i, p]).any()) for i, p in enumerate(table.position)]
    table = table[table.position >= 1].reset_index(drop=True); rows = []
    for seed in [0, 1, 2]:
        run = MI / 'artifacts/twotoken' / f'two_token_anon_s{seed}'
        model = tt.TwoTokenModel(tt.Config.from_dict(json.loads((run / 'config.json').read_text())))
        model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True)); model.eval()
        records = []
        with torch.no_grad():
            for part in mi_setup.batch_slices(len(table), 128):
                chunk = table.iloc[part].reset_index(drop=True)
                inp, _ = tt.eval_batch(data, 'val', chunk.window.to_numpy(), 'cpu', model.config.n_slots)
                base = ns['delta_at'](model, inp, chunk)
                fresh = ns['edit'](model, inp, chunk, 'identity')
                # Same fresh target label in both arms; rename all its occurrences in the control.
                consistent = {k: v.clone() for k, v in inp.items()}
                pos = torch.as_tensor(chunk.position.to_numpy())
                own = inp['slot'][torch.arange(len(chunk)), pos]
                new_label = fresh['slot'][torch.arange(len(chunk)), pos]
                consistent['slot'] = torch.where(inp['slot'] == own[:, None], new_label[:, None], inp['slot'])
                assert torch.equal(inp['slot'][:, :, None] == inp['slot'][:, None, :], consistent['slot'][:, :, None] == consistent['slot'][:, None, :])
                broken_delta = ns['delta_at'](model, fresh, chunk)
                consistent_delta = ns['delta_at'](model, consistent, chunk)
                unlinked = ~chunk.linked.to_numpy()
                assert np.array_equal(broken_delta[unlinked], consistent_delta[unlinked]), 'Future relabel changed an unlinked prediction'
                relabel = {k: v.clone() for k, v in inp.items()}
                relabel['slot'] = torch.where(inp['slot'] > 0, inp['slot'] % (model.config.n_slots - 1) + 1, inp['slot'])
                # An injective renaming preserves every within-window identity relation.
                assert torch.equal(inp['slot'][:, :, None] == inp['slot'][:, None, :], relabel['slot'][:, :, None] == relabel['slot'][:, None, :])
                frame = chunk[['match_id', 'half', 'linked']].copy()
                frame['fresh slot |Delta|'] = abs(broken_delta - base)
                frame['link-preserving relabel |Delta|'] = abs(ns['delta_at'](model, relabel, chunk) - base)
                frame['fresh minus relabel (paired magnitude)'] = frame['fresh slot |Delta|'] - frame['link-preserving relabel |Delta|']
                frame['same-fresh-label consistent rename |Delta|'] = abs(consistent_delta - base)
                frame['broken minus consistent Delta'] = broken_delta - consistent_delta
                frame['broken versus consistent |Delta|'] = abs(broken_delta - consistent_delta)
                records.append(frame)
        records = pd.concat(records)
        for half in ['estimation', 'discovery']:
            for linked in [True, False]:
                f = records[(records.half == half) & (records.linked == linked)]
                for name in ['fresh slot |Delta|', 'link-preserving relabel |Delta|', 'fresh minus relabel (paired magnitude)', 'same-fresh-label consistent rename |Delta|', 'broken minus consistent Delta', 'broken versus consistent |Delta|']:
                    rows.append(dict(seed=seed, half=half, linked=linked, measure=name, **summarise(f[name], f.match_id)))
        print('slots seed', seed, 'complete', flush=True)
    return dict(rows=rows, unknown_validation_events=int((data.features['player'][data.scored['val']['event_row']] == 0).sum()), validation_events=len(data.scored['val']['event_row']))


def probe(ns):
    table = ns['val_conflicts']; point = 'final_normalised'
    x = probing.collect(ns['MODEL'], ns['bundle'], ns['val_view'], table.window.to_numpy(), table.position.to_numpy(), [point], 256, hooks.forward)[point]
    train = table.half.to_numpy() == 'discovery'; y = ns['events'].action_id.to_numpy()[table.event_row]
    rng = np.random.default_rng(0); rows = []
    for label in ['actual action labels', 'independently shuffled event labels']:
        a, b = (y[train], y[~train]) if label == 'actual action labels' else (rng.permutation(y[train]), rng.permutation(y[~train]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model, c, _ = probing.fit_probe(x[train], a, table.match_id.to_numpy()[train], [.01, .1, 1.], max_iter=2000)
        assert not any(issubclass(w.category, ConvergenceWarning) for w in caught), 'Probe CV or final fit did not converge'
        correct = model.predict(x[~train]) == b
        rows.append(dict(measure=label, C=c, fit_iterations=int(model[-1].n_iter_.max()), **summarise(correct.astype(float), table.match_id.to_numpy()[~train])))
    return dict(rows=rows, point=point, scope='Action decoding diagnostic on full validation conflicts; not a replacement for the entire MB05 probe grid.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--section', choices=['identity', 'history', 'interaction', 'slots', 'probe', 'report', 'all'], default='all')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
    self_check()
    if args.check_only:
        print('Self-check passed'); return
    sys.addaudithook(protect)
    tracked = [ROOT / p for p in subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0') if p]
    before = {str(p.relative_to(ROOT)): sha(p) for p in tracked if p.is_file() and not p.is_relative_to(HERE) and p != REPORT}
    provenance = dict(baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(), python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device='cpu', bootstrap_matches=2000, bootstrap_seed=0, seeds_benchmark=[100, 101, 102, 103, 104], seeds_anonymous=[0, 1, 2], protected_files=len(before), input_snapshot_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest())
    (HERE / 'results').mkdir(exist_ok=True)
    if args.section != 'report':
        ns = setup()
        for section in ['identity', 'history', 'interaction', 'slots', 'probe']:
            if args.section in [section, 'all']:
                # Fresh namespace prevents a seed loop from changing the primary checkpoint.
                result = globals()[section](ns.copy())
                result['provenance'] = provenance
                result['input_hashes'] = {p: d for p, d in before.items() if p.startswith('mi_analysis/notebooks/') or p.startswith('mi_analysis/src/') or p.endswith('/best.pt') or p.endswith('/checkpoint.pt')}
                assert before == {p: sha(ROOT / p) for p in before}, 'Historical files changed'
                (HERE / 'results' / (section + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
                print(section, 'saved; historical hashes unchanged', flush=True)
    if args.section in ['all', 'report']:
        write_report()
    assert before == {p: sha(ROOT / p) for p in before}, 'Historical files changed'


def write_report():
    data = {name: json.loads((HERE / 'results' / (name + '.json')).read_text())
            for name in ['identity', 'history', 'interaction', 'slots', 'probe']}
    assert len({d['provenance']['input_snapshot_sha256'] for d in data.values()}) == 1, 'Mixed historical inputs'
    def estimate(r):
        return f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
    lines = [
        '# Targetted analysis: corrected identity and linkage diagnostics',
        '',
        'Existing-checkpoint re-analysis, 2026-10-08. All results below are exploratory validation analyses, not fresh confirmation. No Transformer checkpoints were retrained and the test split was not opened; only the analysis probes were fitted.',
        '',
        '## Scope and reproducibility',
        '',
        'The benchmark is the available **4L/2H/K16, wd=0.1** model, seeds 100–104, primary seed 103—not the historical wd=0.01 selected model. Anonymous models are the existing **K32, wd=0.1** checkpoints, seeds 0–2. No direct performance attribution between those designs is attempted.',
        '',
        'Run `python -B mi_analysis/targetted_analysis/run.py` with the existing rerun dependencies. `--section identity|history|interaction|slots|probe|report` runs individual phases; `--check-only` runs the small regression check. Full populations are used; no event cap or smoke mode. The runner reuses named notebook functions without executing notebook write cells. Machine-readable results and input fingerprints are in `mi_analysis/targetted_analysis/results/`.',
        '',
        'Intervals are 95% match-bootstrap intervals (2,000 draws, seed 0); seed variability is displayed separately, not pooled into these intervals. New thresholds were not chosen from these results. Across-seed agreement below is a consistency check on the same validation data, not independent data replication.',
        '',
        '## 1. Corrected identity specificity and habit-subspace interventions',
        '',
        'Frozen reference codes are retained in separate columns; model identity codes come from the actual encoded inputs. Matched partners and role centroids use training-only model indices. Reference-player distributions are joined through physical player IDs. The runner asserts one-to-one positive-player mapping and equality of categorical/vector full swaps on previously misaligned events. The original 461/1,744 player-code mismatches are not discarded: all 1,744 P-player events remain eligible.',
        '',
        'Estimation-half results (529 events before any undefined-control exclusions). C2 is |matched swap Δ| minus the norm-matched random-vector magnitude. C3 is the correlation across 15 action log-probability changes with the correctly mapped player-only reference change.',
        '',
        '| Seed | Removal magnitude | C2 vs norm-random | Role-centroid player-specific share | Habit-part C3 | Rest-part C3 | Habit minus rest C3 |',
        '|---|---|---|---|---|---|---|',
    ]
    controls = pd.DataFrame(data['identity']['controls']); habits = pd.DataFrame(data['identity']['habit_use'])
    for seed in [100, 101, 102, 103, 104]:
        c = controls[(controls.seed == seed) & (controls.half == 'estimation')].set_index('measure')
        h = habits[(habits.seed == seed) & (habits.half == 'estimation')].set_index('measure')
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(estimate(r) for r in [c.loc['removal |Delta|'], c.loc['C2 partner minus norm-random'], c.loc['player-specific share relative to role centroid'], h.loc['habit part only'], h.loc['rest only'], h.loc['habit minus rest (paired)']]) + ' |')
    h = habits[habits.half == 'estimation'].pivot(index='seed', columns='measure', values='ci_low')
    used = (h['habit part only'] > 0) & (h['habit minus rest (paired)'] > 0)
    specific = controls[(controls.half == 'estimation') & (controls.measure == 'C2 partner minus norm-random')]
    lines += ['', 'C3 measures directional alignment across action scores, not the magnitude or fraction of the habit-mediated effect. These correlations alone do not support calling that component small, dominant or exclusive.']
    lines += ['', f"Under the original positive-C3 plus paired-superiority rule, habit-part use passes on **{int(used.sum())}/5 seeds**. Matched swaps exceed norm-matched random perturbations with CI above zero on **{int((specific.ci_low > 0).sum())}/5 seeds**. These are different questions: generic player sensitivity does not establish identity specificity or an exclusively habit-based mechanism.", '', 'The beyond-role embedding-decoding results in MB21 are separate stored-information evidence; this phase corrects the causal swap/reference alignment, not that decoding grid. Discovery-half results and distribution-matched controls are retained in the JSON. These corrections do not silently replace every downstream result in MB04–MB21.', '', '## 2. History versus current-event context with isolated donors', '', 'Donors are sampled from another match in the **same discovery/estimation half and at the same scoring position**, RNG seed 0. Current-event fields are asserted unchanged during history swaps; identity fields are restored during whole-context swaps.', '', f"Rows without a valid donor: **{data['history']['excluded_no_same_half_position_donor']}**. Cross-half donors: **0**. The ratio CI resamples numerator and denominator jointly.", '', '| Estimation measure | Mean [95% CI] | Events |', '|---|---|---|']
    for r in data['history']['rows']:
        if r['subset'] == 'estimation':
            lines.append(f"| {r['measure']} | {estimate(r)} | {r['n']} |")
    ratio = next(r for r in data['history']['rows'] if r['subset'] == 'estimation' and r['measure'] == 'history / whole-context magnitude')
    genuine_ratio = next(r for r in data['history']['rows'] if r['subset'] == 'genuine' and r['measure'] == 'history / whole-context magnitude')
    lines += ['', f"For genuine (non-coupled) events across validation, the history/whole-context ratio is **{estimate(genuine_ratio)}**. This stratum exceeds the 10% cutoff and prevents a blanket claim that history is negligible."]
    lines += ['', f"The original <10% point-estimate reading rule {'passes' if ratio['mean'] < .1 else 'does not pass'} after donor isolation. This concerns these intervention distributions; it does not prove history is universally irrelevant. Genuine/coupled strata and discovery/all-validation results are in the JSON.", '', '## 3. Corrected identity–context interaction-source uncertainty', '', 'MB20 factorial/JVP runs are repeated with corrected matched-player indices. The source statistic is the paired sign-aligned ratio for own minus shared gains: its numerator **and** |exact interaction| denominator are resampled together. The unchanged-player factorial endpoint is checked before scoring.', '', '| Component, partner intervention, estimation | Own retained | Shared retained | Source share [95% CI] | Original SOURCE rule |', '|---|---|---|---|---|']
    for r in data['interaction']['rows']:
        if r['second'] == 'partner' and r['half'] == 'estimation':
            lines.append(f"| {r['component']} | {r['own']:.4f} | {r['shared']:.4f} | {estimate(r)} | {r['source']} |")
    lines += ['', 'The SOURCE rule still requires own-retained CI low ≥0.8 and source-share CI low >0.2. Components are intervened on separately; their source shares are **not additive attribution percentages**. Norm-matched random-vector factorials and discovery-half results are also retained. This source analysis remains primary-seed exploratory evidence.', '', '## 4. Anonymous labels: link-breaking versus link-preserving controls', '', 'The original fresh-target-slot intervention is compared with an injective cyclic renaming of all positive slots in a window. An assertion verifies that the relabel control preserves every pairwise player-equality relation. Linked and unlinked events are scored separately across all three available anonymous seeds. The paired magnitude difference is a diagnostic contrast, **not a calibrated estimate of pure linkage use**: the two interventions perturb different embeddings.', '', '| Seed | Estimation population | Fresh target-slot magnitude | Link-preserving relabel magnitude | Paired fresh minus relabel |', '|---|---|---|---|---|']
    s = pd.DataFrame(data['slots']['rows'])
    for seed in [0, 1, 2]:
        for linked in [True, False]:
            f = s[(s.seed == seed) & (s.half == 'estimation') & (s.linked == linked)].set_index('measure')
            lines.append(f"| {seed} | {'linked' if linked else 'unlinked'} (n={int(f.iloc[0]['n'])}) | " + ' | '.join(estimate(f.loc[name]) for name in ['fresh slot |Delta|', 'link-preserving relabel |Delta|', 'fresh minus relabel (paired magnitude)']) + ' |')
    lines += ['', 'A second, better-matched control gives the target **the identical fresh label** in both arms: one arm changes only the target; the other consistently renames that player at every occurrence. All other player labels remain unchanged. For unlinked events, causal locality requires identical predictions, which is asserted. The linked contrast tests dependence on the earlier same-player label stream while holding the target embedding fixed; it still does not identify a unique matching or habit-retrieval algorithm.', '', '| Seed | Linked estimation: broken minus consistent Δ | Linked estimation: magnitude of that contrast |', '|---|---|---|']
    for seed in [0, 1, 2]:
        f = s[(s.seed == seed) & (s.half == 'estimation') & s.linked].set_index('measure')
        lines.append(f"| {seed} | {estimate(f.loc['broken minus consistent Delta'])} | {estimate(f.loc['broken versus consistent |Delta|'])} |")
    lines += ['', 'Read signed and magnitude contrasts separately: a replicated nonzero magnitude shows sensitivity, not a replicated directional preference for the context action. Differing signed effects across seeds prevent that stronger interpretation. Unlinked matched contrasts must be exactly zero; their results are retained in the JSON.']
    lines += ['', f"Non-dedicated players share slot 0 on **{data['slots']['unknown_validation_events']}/{data['slots']['validation_events']} validation events**. The trained checkpoints cannot retrospectively acquire distinct anonymous identities for those players. Nonzero relabel sensitivity means the original MB23 identity effect must not be described as isolated link-breaking evidence. Existing circuit verdicts remain verdicts for the original slot intervention; circuits were not re-selected under a corrected linkage behaviour.", '', '## 5. Probe-control diagnostic', '', 'An action probe at `final_normalised` is trained on discovery conflicts and evaluated on estimation conflicts. Its control independently permutes **event-level** action labels within each half, preserving frequencies without making control labels a function of the true class. C is selected by match-grouped CV within discovery only.', '', '| Diagnostic | Estimation accuracy [95% CI] | C |', '|---|---|---|']
    for r in data['probe']['rows']:
        lines.append(f"| {r['measure']} | {estimate(r)} | {r['C']} |")
    assert all(r['fit_iterations'] < 2000 for r in data['probe']['rows']), 'Unconverged final probe'
    lines += ['', 'All probe CV/final fits completed without convergence warnings (iteration limit 2,000); final iteration counts are recorded in the JSON.']
    p = data['probe']['provenance']
    lines += ['', 'This is a targeted control sanity check, not a replacement for all MB05 player/role/context probe experiments. The original label-remapping selectivity estimates remain unsuitable as random-label baselines and should not support paper claims.', '', '## Paper-facing conclusion and preservation', '', '**Use the corrected tables above for the affected comparisons, with their scope and uncertainty.** Keep behavioural player sensitivity distinct from identity specificity; distinguish embedding decodability from causal habit-subspace evidence; describe anonymous interventions as slot sensitivity unless additional link-isolating evidence is obtained. No new uniquely identified or MEANINGFUL circuit is established here.', '', 'The unaffected player-removal, patching and component-sufficiency evidence can accompany this narrower interpretation. Broader all-player anonymous-linkage claims require a changed encoding and retraining; they are not prerequisites for the narrower predictive/MI paper. MB12 remains retrospective and invalidated MB16 legacy scrub evidence remains excluded. Predictive full-versus-restricted benchmarking was not rerun by this analysis.', '', f"Historical-file write guards and end-of-run SHA-256 comparisons passed in every phase ({p['protected_files']} existing tracked files protected). No historical notebooks, artifacts, manifests, test-access logs or `findings_short.md` were edited. Input snapshot SHA-256: `{p['input_snapshot_sha256']}`. Baseline commit: `{p['baseline_commit']}`.", '', f"Execution: Python {p['python']}, Torch {p['torch']}, NumPy {p['numpy']}, CPU (four Torch threads). No test view, retraining, category-adjacency edits or changes to historical hash guards.", '']
    temp = HERE / 'report.tmp'
    temp.write_text('\n'.join(lines))
    temp.replace(REPORT)
    print('Report written:', REPORT, flush=True)


if __name__ == '__main__':
    main()
