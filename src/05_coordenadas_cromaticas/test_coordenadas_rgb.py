import unittest

import numpy as np
import pandas as pd

from coordenadas_RGB import RAIZ, calcular_coordenadas, coordenadas_globais, numero_camadas

ATOL = 1e-12


class TestCoordenadas(unittest.TestCase):
    def test_primarias_cinza_zero_e_entradas_intactas(self):
        canais = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100],
                           [100, 100, 100], [0, 0, 0]]).T
        original = canais.copy()
        r, g, b, _, validos = calcular_coordenadas(*canais)
        np.testing.assert_array_equal(canais, original)
        np.testing.assert_array_equal(validos, [True] * 4 + [False])
        esperado = [*np.eye(3), [1 / 3] * 3, [np.nan] * 3]
        np.testing.assert_allclose(np.column_stack((r, g, b)), esperado, atol=ATOL, rtol=0)

    def test_formas_incompativeis(self):
        with self.assertRaisesRegex(ValueError, "mesma forma"):
            calcular_coordenadas([1, 2], [[1, 2]], [1, 2])

    def test_valores_invalidos(self):
        r, g, b, _, validos = calcular_coordenadas(
            [np.nan, 1, 1, -1, 1e308], [1, np.inf, 1, 2, 1e308], [1, 1, -np.inf, 1, 0]
        )
        self.assertFalse(validos.any())
        self.assertTrue(np.isnan([r, g, b]).all())
        for limiar in (-1, np.nan, np.inf):
            with self.subTest(limiar=limiar), self.assertRaises(ValueError):
                calcular_coordenadas([1], [1], [1], limiar)

    def test_limiar_estrito(self):
        r, g, b, soma, validos = calcular_coordenadas([0, 5, 6], [0]*3, [0]*3, 5)
        np.testing.assert_array_equal(validos, [False, False, True])
        self.assertTrue(np.isnan([r[:2], g[:2], b[:2]]).all())
        self.assertTrue(np.isnan(coordenadas_globais(soma, soma, soma, validos & False)).all())

    def verificar_laser(self, laser):
        pasta = RAIZ / "resultados/analise_espacial" / laser / f"resultados_espaciais_{laser}_eixos"
        momentos = pd.read_csv(pasta / "momentos_por_imagem_e_canal.csv")
        integrado = momentos.pivot(index=["arquivo", "camadas"], columns="canal", values="P")
        arquivos = sorted((pasta / "dados_I_xy").glob("I_xy_*.npz"))
        self.assertTrue(arquivos)
        self.assertEqual(
            {(p.stem.removeprefix("I_xy_") + ".png", numero_camadas(p.stem)) for p in arquivos},
            set(integrado.index),
        )
        for arquivo in arquivos:
            with self.subTest(arquivo=arquivo.name), np.load(arquivo, allow_pickle=False) as dados:
                canais = [dados[f"I_{c}"] for c in "RGB"]
                r, g, b, soma, validos = calcular_coordenadas(*canais, limiar=0)
                self.assertTrue(validos.any())
                rgb = np.array([r[validos], g[validos], b[validos]])
                np.testing.assert_allclose(rgb.sum(axis=0), 1, atol=ATOL, rtol=0)
                self.assertTrue(((rgb >= -ATOL) & (rgb <= 1 + ATOL)).all())
                self.assertTrue(np.isnan([r[~validos], g[~validos], b[~validos]]).all())
                chave = (arquivo.stem.removeprefix("I_xy_") + ".png", numero_camadas(arquivo.stem))
                p = integrado.loc[chave, list("RGB")].to_numpy()
                np.testing.assert_allclose(
                    coordenadas_globais(*canais[:2], soma, validos), p / p.sum(), atol=ATOL, rtol=0
                )

    def test_laser_azul(self):
        self.verificar_laser("azul")

    def test_laser_verde(self):
        self.verificar_laser("verde")


if __name__ == "__main__":
    unittest.main()
