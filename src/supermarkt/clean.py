import re
from dataclasses import dataclass

@dataclass
class Product:
    """
    Data structure representing a supermarket product and its attributes.
    """

    href: str | None
    sub_cat: str | None
    name: str
    price: float
    quantity: str
    ingredients: str | None
    nutrients: dict | None

    kcal: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    protein_g: float | None = None
    salt_g: float | None = None

def clean_product_dirk(p: Product):
    """
    Clean and standardize product attributes for Dirk products.
    """
    if p.ingredients is not None:
        p.ingredients = re.sub(r"^Ingrediënten\s*:\s*", "", p.ingredients)
    
    if p.nutrients is not None:
        # Parse the individual nutrients
        for nutrient, amount in p.nutrients.items():
            if "kcal" in nutrient:
                p.kcal = float(amount)
            elif "Koolhyd" in nutrient:
                p.carbs_g = float(amount)
            elif "Vet" in nutrient:
                p.fat_g = float(amount)
            elif "Eiwit" in nutrient:
                p.protein_g = float(amount)
            elif "Zout" in nutrient:
                p.salt_g = float(amount)