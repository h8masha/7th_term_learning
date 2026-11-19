package ru.tvp.lr1;

import java.util.LinkedHashMap;
import java.util.Map;


public class Labwork1 {

    public static LinkedHashMap<Long, Long> pairs = new LinkedHashMap<Long, Long>();

    static {
        pairs.put(5L, 3L);
        pairs.put(10L, 7L);
        pairs.put(20L, 15L);
        pairs.put(50L, 40L);
        pairs.put(100L, 80L);
        pairs.put(200L, 150L);
    }
    // время если че в милисекундах
    public static String comparison(Map<Long, Long> pairs) throws RuntimeException{
        StringBuilder result = new StringBuilder("x    y    result    timeIter    timeRecurs\n");

        for (Long x : pairs.keySet()) {
            long y = pairs.get(x);
            long startTime = System.nanoTime();
            long resultIteration = iterationMethod(x, y);
            long endTime1 = System.nanoTime();
            long resultRecursive = recursionFunction(x, y);
            long endTime2 = System.nanoTime();
            double iterationTime = (endTime1 - startTime) / 1000000.0;
            double recursionTime = (endTime2 - endTime1) / 1000000.0;
            if (resultIteration != resultRecursive) {
                throw new RuntimeException("Results of pair: " + x + " " + y + " are not equals in 2 methods");
            }
            result.append(x)
                    .append("    ")
                    .append(y).append("    ")
                    .append(resultIteration).append("   ")
                    .append(iterationTime)
                    .append("    ")
                    .append(recursionTime).append("\n");
        }
        return result.toString();
    }

    public static void main(String ... args) {
        try {
            System.out.println(comparison(pairs));
        }
        catch(Exception e) {
            System.out.println(e.getMessage());
        }
    }
    //через итеративную реализацию
    public static long iterationMethod(long x, long y) throws IllegalArgumentException{
        if (x < 0 || y < 0) {
            throw new IllegalArgumentException("Incorrect input type");
        }

        long product = 0;
        for (long i = 0; i < x; i++) {
            product += y;
        }

        long doubleX = 0;
        for (long i = 0; i < x; i++) {
            doubleX += 2;
        }

        return product + doubleX;
    }

    //через рекурсивную реализацию (обработка переполнения - try-catch)
    public static long recursionAdd(long a, long b) {
        if (a == 0) {
            return b;
        }

        return 1 + recursionAdd(a - 1, b);
    }

    public static long recursionMultiply(long a, long b) {
        if (a == 0) {
            return 0;
        }

        return recursionAdd(recursionMultiply(a - 1, b), b);
    }

    public static long recursionDouble(long n) {
        if (n == 0) {
            return 0;
        }

        return 2 + recursionDouble(n - 1);
    }

    public static long recursionFunction(long x, long y) throws RuntimeException{
        try {
            return recursionAdd(recursionMultiply(x, y), recursionDouble(x));
        }
        catch(StackOverflowError e) {
            throw new RuntimeException("Stack was overflowed!");
        }
    }
}