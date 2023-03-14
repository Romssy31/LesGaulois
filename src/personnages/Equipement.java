package personnages;

public enum Equipement {
    CASQUE,
    BOUCLIER
}

public class Test{
	public static void main(String[] args) {
	        Equipement equip1 = Equipement.CASQUE;
	        Equipement equip2 = Equipement.BOUCLIER;
	        
	        System.out.println("Le premier équipement est un " + equip1);
	        System.out.println("Le deuxième équipement est un " + equip2);
	    }
}
