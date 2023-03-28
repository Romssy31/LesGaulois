package personnages;

public class Romain {
	private String nom;
	private int force;
	private int force1;
	private int force2;
	
	
	public Romain(String nom, int force) {
		this.nom = nom;
		this.force = force;
		assert(this.force>=0):"Invalider user force";
	}
	public String getNom() {
		return nom;
	}
	
	public void parler(String texte) {
		System.out.println(prendreParole() + "« " + texte + "»");
	}
	
	private String prendreParole() {
		return "Le romain " + nom + " : ";
	}
	
	public void recevoirCoup(int forceCoup) {
		assert(this.force>0):"invalide force entry";
		force1 = force;
		force -= forceCoup;
		force2 = force;
		assert(force1>force2):"post-condition";
		if (force > 0) {
			parler("Aïe");
		} else {
			parler("J'abandonne...");
		}
	}
	
	public static void main1(String[] args) {
		
		Romain minus = new Romain("Minus", 6);
		System.out.println(minus.prendreParole());
		minus.parler("Je suis Minus !");
		minus.recevoirCoup(9);
		}
	
	public static void main(String[] args) {
		System.out.println(Equipement.CASQUE);
		System.out.println(Equipement.BOUCLIER);
		Romain minus = new Romain("Minus", 2);
		Equipement casque = Equipement.CASQUE;
		Equipement bouclier = Equipement.BOUCLIER;
		
		minus.sEquiper(casque);
		minus.sEquiper(casque);
		minus.sEquiper(bouclier);
		minus.sEquiper(casque);
	}
	
	private Equipement[] equipements = new Equipement[2];
	private int nbEquipement = 0;

	
	public void sEquiper(Equipement e) {
		switch (nbEquipement){
		case 0:
			equipements[0] = e;
			nbEquipement++;
			System.out.println("Le soldat " + nom + " s'equipe avec un " + e.getNom() + ".");
			break;
		case 1:
			if (equipements[0] == e) {
				System.out.println("Le soldat " + nom + " posséde déja " + e.getNom() + ".");
			}
			else {
				equipements[1] = e;
				nbEquipement++;
				System.out.println("Le soldat " + nom + " s'equipe avec un " + e.getNom() + ".");
			}break;
			
		case 2:
			System.out.println("Le soldat" + nom + " est déjà bien protégé !");
				
			}
		}
	}