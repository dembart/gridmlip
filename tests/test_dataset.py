"""Dataset building blocks, the recipe/CLI, and reproducibility of the random step."""
import warnings

import numpy as np
import pytest

pytest.importorskip("dscribe")
pytest.importorskip("fpsample")
pytest.importorskip("vesin")
pytest.importorskip("sklearn")
pytest.importorskip("ions")

from ase.io import read  # noqa: E402

from gridmlip.cli import main  # noqa: E402
from gridmlip.dataset import load_settings, rattle, select_fps, soap_features  # noqa: E402
from gridmlip.dataset.recipe import merge_settings, run  # noqa: E402
from gridmlip.defects import frenkel_pairs, interstitials, vacancies  # noqa: E402

from structures import licoo2, lifepo4  # noqa: E402

FAST = {
    "specie": 3,
    "defects": {"resolution": 0.6, "r_min": 1.5, "r_cut": 5.0},
    "selection": {"max_samples": 15},
}


def _materials():
    a, b = lifepo4(), licoo2()
    a.info["material_id"], b.info["material_id"] = "lfp", "lco"
    return [a, b]


def _same(a_list, b_list):
    assert len(a_list) == len(b_list)
    for a, b in zip(a_list, b_list):
        np.testing.assert_array_equal(a.numbers, b.numbers)
        np.testing.assert_allclose(a.positions, b.positions, rtol=0, atol=1e-8)
        np.testing.assert_allclose(a.cell[:], b.cell[:], rtol=0, atol=1e-8)


@pytest.fixture(scope="module")
def cfgs():
    atoms = lifepo4()
    atoms.info["material_id"] = "lfp"
    return interstitials(atoms, 3, resolution=0.6, r_min=1.5, r_cut=5.0)


# ------------------------------------------------------------------ rattle

def test_rattle_is_reproducible(cfgs):
    c = cfgs[3]
    a, b = rattle(c, seed=42), rattle(c.copy(), seed=42)
    _same([a], [b])
    assert not np.allclose(a.positions, c.positions)
    assert a.info["rattle_seed"] == 42 and a.info["rattle_copy"] == 0 and a.info["rattle_key"]


def test_rattle_streams_differ_by_seed_copy_and_material(cfgs):
    c = cfgs[3]
    ref = rattle(c, seed=42).positions
    assert not np.allclose(ref, rattle(c, seed=43).positions)
    assert not np.allclose(ref, rattle(c, seed=42, copy=1).positions)
    other = c.copy()
    other.info["material_id"] = "something-else"
    assert not np.allclose(ref, rattle(other, seed=42).positions)


def test_rattle_key_reproduces_gridmlip_027(cfgs):
    """key=j reproduces the 0.2.7 pipeline: _make_rng(master_seed, material_id, j) + rattle_atoms."""
    from gridmlip.dataset._legacy import _make_rng
    from gridmlip.perturb import rattle_atoms

    c = cfgs[5]
    old = rattle_atoms(c, rng=_make_rng(42, "lfp", 5), d_min=1.0)
    _same([rattle(c, seed=42, key=5)], [old])


def test_rattle_tolerates_input_below_d_min():
    """An ion placed at r_min < d_min used to make every attempt fail (no position noise at all)."""
    atoms = lifepo4()
    atoms.info["material_id"] = "lfp"
    cands = interstitials(atoms, 3, resolution=0.4, r_min=1.0, r_cut=5.0)
    dists = [c.get_distances(len(c) - 1, range(len(c) - 1), mic=True).min() for c in cands]
    c = cands[int(np.argmin(dists))]
    assert min(dists) < 1.4
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        r = rattle(c, seed=1, d_min=1.4, strain_std=0.0)
    assert not np.allclose(r.positions, c.positions)


def test_rattle_switches(cfgs):
    c = cfgs[0]
    still = rattle(c, seed=1, pos_std=0.0, strain_std=0.0)
    np.testing.assert_allclose(still.positions, c.positions)
    np.testing.assert_allclose(still.cell[:], c.cell[:])


# -------------------------------------------------- features and selection

def test_soap_and_select_edge_cases(cfgs):
    assert soap_features([]).shape == (0, 0)
    X = soap_features(cfgs[:4])
    assert X.shape[0] == 4
    assert len(select_fps(np.empty((0, 5)))) == 0
    assert list(select_fps(X[:2])) == [0, 1]
    assert len(select_fps(X, max_samples=100)) <= 4


def test_select_fps_matches_027_pipeline(cfgs):
    from sklearn.decomposition import PCA
    from gridmlip.select import fps_with_stop

    X = soap_features(cfgs)
    old = fps_with_stop(PCA(n_components=0.95, svd_solver="full").fit_transform(X),
                        rel_tol=1e-5, start_idx="centroid", max_samples=min(20, len(X)))
    np.testing.assert_array_equal(select_fps(X, max_samples=20), old)


# ------------------------------------------------------------ defect labels

def test_parent_labels_do_not_leak_into_configurations():
    atoms = lifepo4()
    atoms.info.update(material_id="lfp", energy=-123.4, REF_energy=-1.0, stress=np.zeros(6))
    atoms.arrays["forces"] = np.ones((len(atoms), 3))
    for func in (interstitials, vacancies, frenkel_pairs):
        c = func(atoms, 3, resolution=0.6, r_min=1.5, r_cut=5.0)[0]
        assert c.info["material_id"] == "lfp"
        assert not {"energy", "REF_energy", "stress"} & set(c.info)
        assert "forces" not in c.arrays and c.calc is None


# ------------------------------------------------------ recipe and the CLI

def test_readme_loop_equals_command(tmp_path):
    settings = merge_settings(FAST)
    run(_materials(), tmp_path, settings, progress=False)

    for atoms in _materials():
        p = dict(resolution=0.6, r_min=1.5, r_cut=5.0, symprec=0.1)
        cfgs = (interstitials(atoms, 3, **p)
                + vacancies(atoms, 3, max_sep_dist="auto", upper_bound=6.0, **p)
                + frenkel_pairs(atoms, 3, max_sep_dist="auto", upper_bound=6.0, **p))
        X = soap_features(cfgs, center=-1, r_cut=6.0, n_max=3, l_max=2, sigma=0.3)
        idx = select_fps(X, pca=0.95, max_samples=15, rel_tol=1e-5)
        train = [rattle(cfgs[j], seed=42, pos_std=0.05, strain_std=0.02, d_min=1.0) for j in idx]
        _same(train, read(tmp_path / f"dataset_{atoms.info['material_id']}.xyz", index=":"))


def test_order_and_chunks_do_not_change_results(tmp_path):
    settings = merge_settings(FAST)
    run(_materials(), tmp_path / "full", settings, progress=False)
    run(_materials()[::-1], tmp_path / "reversed", settings, progress=False)
    run(_materials(), tmp_path / "chunks", settings, chunk=(0, 2), progress=False)
    run(_materials(), tmp_path / "chunks", settings, chunk=(1, 2), progress=False)
    for mid in ("lfp", "lco"):
        ref = read(tmp_path / "full" / f"dataset_{mid}.xyz", index=":")
        _same(ref, read(tmp_path / "reversed" / f"dataset_{mid}.xyz", index=":"))
        _same(ref, read(tmp_path / "chunks" / f"dataset_{mid}.xyz", index=":"))


def test_cli_config_roundtrip_resume_and_dry_run(tmp_path):
    from ase.io import write
    import yaml

    write(tmp_path / "in.xyz", _materials())
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(FAST))

    assert main(["dataset", str(tmp_path / "in.xyz"), "--config", str(tmp_path / "cfg.yaml"),
                 "--output", str(tmp_path / "a"), "--quiet"]) == 0
    record = load_settings(tmp_path / "a" / "config.yaml")
    assert record["seed"] == 42 and "gridmlip" in record["versions"]

    # rerun from the written record: identical files
    assert main(["dataset", str(tmp_path / "in.xyz"), "--config", str(tmp_path / "a" / "config.yaml"),
                 "--output", str(tmp_path / "b"), "--quiet"]) == 0
    for mid in ("lfp", "lco"):
        _same(read(tmp_path / "a" / f"dataset_{mid}.xyz", index=":"),
              read(tmp_path / "b" / f"dataset_{mid}.xyz", index=":"))

    # resume: existing materials are skipped
    main(["dataset", str(tmp_path / "in.xyz"), "--config", str(tmp_path / "cfg.yaml"),
          "--output", str(tmp_path / "a"), "--quiet"])
    assert "exists" in (tmp_path / "a" / "summary.csv").read_text()

    # dry run: counts only
    main(["dataset", str(tmp_path / "in.xyz"), "--config", str(tmp_path / "cfg.yaml"),
          "--output", str(tmp_path / "dry"), "--dry-run", "--quiet"])
    assert not list((tmp_path / "dry").glob("dataset_*.xyz"))
    assert "dry run" in (tmp_path / "dry" / "summary.csv").read_text()


def test_settings_reject_typos():
    with pytest.raises(KeyError, match="pos_sdt"):
        merge_settings({"rattle": {"pos_sdt": 0.03}})


def test_old_pipeline_is_deprecated(tmp_path):
    from gridmlip.dataset import generate_defects_dataset

    with pytest.warns(DeprecationWarning):
        generate_defects_dataset([lifepo4()], output_dir=tmp_path, specie=3, resolution=0.6,
                                 defect_type="int", max_samples=5)
