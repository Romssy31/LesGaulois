package personnages;

public enum Equipement {
    CASQUE,
    BOUCLIER
}

public class Test{
	public static void main(String[] args) {
	        Equipement equip1 = Equipement.CASQUE;
	        Equipement equip2 = Equipement.BOUCLIER;
	        
	        System.out.println("Le premier �quipement est un " + equip1);
	        System.out.println("Le deuxi�me �quipement est un " + equip2);
	    }
}
