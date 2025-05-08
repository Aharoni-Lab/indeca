import itertools as itt
import os
from uuid import uuid4

import pandas as pd
from lxml import etree
from svgutils import transform as _transform
from svgutils.compose import CONFIG, SVG, Panel, Text, Unit


def make_svg_panel(label, im_path, param_text, im_scale=1, fix_mpl=True, sh=None):
    im = SVG(im_path, fix_mpl=fix_mpl)
    if im_scale != 1:
        im = im.scale(im_scale)
    lab = Text(label, **param_text)
    tsize = param_text["size"]
    if sh is None:
        x_sh, y_sh = 0.6 * tsize, 1 * tsize
    else:
        x_sh, y_sh = sh
    pan = Panel(im.move(x=x_sh, y=y_sh), lab.move(x=0, y=tsize))
    pan.height = im.height * im_scale + y_sh
    pan.width = im.width * im_scale + x_sh
    return pan


def svg_unique_id(fname, out_name=None):
    doc = etree.parse(fname)
    rt = doc.getroot()
    nmap = rt.nsmap
    id_eles = doc.findall("//path[@id]", nmap)
    ids = list(set([e.attrib["id"] for e in id_eles]))
    id_dict = {i: str(uuid4())[:8] for i in ids}
    attr_name = "{{{}}}href".format(nmap["xlink"])
    for ele in id_eles:
        ele.attrib["id"] = id_dict[ele.attrib["id"]]
    for ele in doc.findall("//use[@xlink:href]", nmap):
        try:
            ele.attrib[attr_name] = "#" + id_dict[ele.attrib[attr_name][1:]]
        except KeyError:
            pass
    if out_name is None:
        out_name = fname
    doc.write(out_name, pretty_print=True)


class GridSpec(Panel):
    def __init__(
        self,
        param_text=None,
        width=None,
        height=None,
        hsep=0,
        wsep=0,
        halign="center",
        valign="center",
        **elements,
    ):
        ele_df = []
        for ename, ele in elements.items():
            edat = ele[0]
            row, col = ele[1]
            try:
                ew, eh = ele[0].width, ele[0].height
            except AttributeError:
                edat = make_svg_panel(ename, edat, param_text)
                ew, eh = edat.width, edat.height
            try:
                rspan, cspan = ele[2]
            except IndexError:
                rspan, cspan = 1, 1
            ele_df.append(
                {
                    "name": ename,
                    "dat": edat,
                    "row": row,
                    "col": col,
                    "rowspan": rspan,
                    "colspan": cspan,
                    "height": eh,
                    "width": ew,
                }
            )
        self._elements = pd.DataFrame(ele_df)
        super().__init__(*self._elements["dat"])
        self.width = width
        self.height = height
        self.hsep = hsep
        self.wsep = wsep
        self.halign = halign
        self.valign = valign

    def save(self, fname):
        """Save figure to SVG file.

        Parameters
        ----------
        fname : str
            Full path to file.
        """
        element = _transform.SVGFigure(self.width, self.height)
        element.append(self)
        element.save(os.path.join(CONFIG["figure.save_path"], fname))

    def tile(self):
        self._compute_spc()
        self._compute_tile()
        for idx, el in enumerate(self):
            erow = self._elements.loc[idx]
            if self.halign == "left":
                x = erow["box_x"]
            elif self.halign in ("middle", "center"):
                x = erow["box_x"] + (erow["box_w"] - erow["width"]) / 2
            elif self.halign == "right":
                x = erow["box_x"] + erow["box_w"] - erow["width"]
            else:
                raise ValueError(
                    "halign must be 'left', 'right', 'middle' or 'center', got {} instead".format(
                        self.halign
                    )
                )
            if self.valign == "top":
                y = erow["box_y"]
            elif self.valign in ("middle", "center"):
                y = erow["box_y"] + (erow["box_h"] - erow["height"]) / 2
            elif self.valign == "bottom":
                y = erow["box_y"] + erow["box_h"] - erow["height"]
            else:
                raise ValueError(
                    "valign must be 'top', 'bottom', 'middle' or 'center', got {} instead".format(
                        self.valign
                    )
                )
            el.move(x, y)

    def _compute_spc(self):
        spc_df = []
        for _, ele in self._elements.iterrows():
            for r, c in itt.product(range(ele["rowspan"]), range(ele["colspan"])):
                spc_df.append(
                    {
                        "name": ele["name"],
                        "row": ele["row"] + r,
                        "col": ele["col"] + c,
                        "width": ele["width"] / ele["colspan"],
                        "height": ele["height"] / ele["rowspan"],
                    }
                )
        self._spc_df = pd.DataFrame(spc_df)

    def _compute_tile(self):
        row_h = self._spc_df.groupby("row")["height"].max()
        col_w = self._spc_df.groupby("col")["width"].max()
        for idx, ele in self._elements.iterrows():
            self._elements.loc[idx, "box_x"] = (
                col_w[: ele["col"]].sum() + ele["col"] * self.wsep
            )
            self._elements.loc[idx, "box_y"] = (
                row_h[: ele["row"]].sum() + ele["row"] * self.hsep
            )
            self._elements.loc[idx, "box_w"] = (
                col_w[ele["col"] : ele["col"] + ele["colspan"]].sum()
                + (ele["colspan"] - 1) * self.wsep
            )
            self._elements.loc[idx, "box_h"] = (
                row_h[ele["row"] : ele["row"] + ele["rowspan"]].sum()
                + (ele["rowspan"] - 1) * self.hsep
            )
        if self.height is None:
            self.height = Unit(row_h.sum() + self.hsep * (len(row_h) - 1))
            self.width = Unit(col_w.sum() + self.wsep * (len(col_w) - 1))
