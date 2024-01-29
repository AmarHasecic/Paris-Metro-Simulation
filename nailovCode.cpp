#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
using namespace std;
int main() {
    // Open the first file
    std::ifstream file1("kod.txt");
    if (!file1.is_open()) {
        std::cerr << "Error opening file1.txt\n";
        return 1;
    }

    // Open the second file
    std::ifstream file2("kod2.txt");
    if (!file2.is_open()) {
        std::cerr << "Error opening file2.txt\n";
        return 1;
    }

    int num1, num2;
    int count = 0;
    int te=0,cc=0;
vector<int> a,b;
    // Loop to compare integers until there are no more in either file
    while (file1 >> num1 && file2 >> num2) {
         /*     if (num1 != num2) {
                te++;
            std::cout << "Files differ at position " << count << ": " << num1 << " vs " << num2 << "\n";
            // Add any additional logic for handling differences
            // For example, you might break the loop or store the differences in a vector.

        }
        else if(num1==2147483647)
            cc++;
          //  if(count>200)
            //    break;
        count++;
    }
    */
            a.push_back(num1);
            b.push_back(num2);

    }
    /*
   sort(a.begin(),a.end());
   sort( b.begin(), b.end());
   */
   count=0;
    for(int i=0;i<a.size()&&i<b.size();i++)
    {

         if (a[i] != b[i]) {
                te++;
            std::cout << "Files differ at position " << i << ": " << a[i] << " vs " << b[i] << "\n";
            // Add any additional logic for handling differences
            // For example, you might break the loop or store the differences in a vector.

        }
        else if(a[i]==2147483647)
            cc++;
          //  if(count>200)
            //    break;
        count++;    }


    // Check if one file has more integers than the other
    if (file1 >> num1) {
        std::cout << "File2 is shorter.\n"<<a.size()<<" "<<b.size();
    } else if (file2 >> num2) {
        std::cout << "File1 is shorter.\n"<<a.size()<<" "<<b.size();;
    }
std::cout<< " ukupno rlici " <<te<<" "<<cc<<std::endl;
    // Close the files
    file1.close();
    file2.close();

    return 0;
}