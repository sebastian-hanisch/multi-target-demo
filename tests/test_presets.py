"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt."""

import mtt_constants as C
import mtt_evaluation as ev
import mtt_presets as P


def _settings_from(p):
    return ev.Settings(angle=p["angle"], dt=p["dt"], sigma=p["sigma"], pd=p["pd"], clutter=p["clutter"], q=p["q"], seed=p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, p in C.PRESETS.items():
        assert C.ANGLE_MIN <= p["angle"] <= C.ANGLE_MAX and C.DT_MIN <= p["dt"] <= C.DT_MAX, name
        assert C.SIGMA_MIN <= p["sigma"] <= C.SIGMA_MAX and C.PD_MIN <= p["pd"] <= C.PD_MAX, name
        assert C.CLUTTER_MIN <= p["clutter"] <= C.CLUTTER_MAX and C.Q_MIN <= p["q"] <= C.Q_MAX and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert _settings_from(C.PRESETS["Standard (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_clamps_numbers():
    a = P.SETTING_SPECS["angle_slider"]
    assert P.clamp_value(a, 500.0) == C.ANGLE_MAX and P.clamp_value(a, -3.0) == C.ANGLE_MIN


def test_the_presets_show_what_their_help_text_claims():
    base = C.PRESETS["Standard (Voreinstellung)"]
    assert C.PRESETS["Zeitversatz am Kreuzungspunkt"]["dt"] > base["dt"]
    assert C.PRESETS["Mit Fehlalarmen"]["clutter"] > base["clutter"]
    assert C.PRESETS["Steiler Winkel"]["angle"] > base["angle"]
