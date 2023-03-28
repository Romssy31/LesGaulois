package personnages;

public enum Equipement {
    CASQUE("Casque"), BOUCLIER("Bouclier");
	
	private String nom;
	private Equipement(String nom) {
		this.nom = nom;
	}
	
	String getNom() {
		return nom;
	}
	
	public String toString() {
		return nom;
	}
}

